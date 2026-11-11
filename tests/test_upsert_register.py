"""Тесты upsert_register.py на синтетических реестрах без семейных данных.

Запуск: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.upsert_register import LOCK_NAME, SourceError, UnknownColumnsError, upsert_rows

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
SKILL_ROOT = SCRIPTS.parent
HEADER = ["person_id", "preferred_label", "notes"]
STAMP = "2026-01-02T03:04:05+00:00"


def write_register(path: Path, rows: list[dict[str, str]], *, bom: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def read_register(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_asset_register(path: Path, rows: list[dict[str, str]] = ()) -> None:
    """Пишет реестр с шапкой из одноимённого шаблона в assets/."""
    with (SKILL_ROOT / "assets" / path.name).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


class UpsertRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "person-register.csv"
        write_register(self.path, [{"person_id": "P-9001", "preferred_label": "Пример", "notes": ""}])

    def assert_rejected(self, rows: list[dict[str, object]], error: type[Exception] = ValueError) -> None:
        before = digest(self.path)
        with self.assertRaises(error):
            upsert_rows(self.path, rows)
        self.assertEqual(digest(self.path), before)

    def test_adds_new_row_in_header_order(self) -> None:
        result = upsert_rows(self.path, [{"notes": "новая", "person_id": "P-9002"}])
        self.assertEqual(result.added, ("P-9002",))
        with self.path.open(encoding="utf-8") as handle:
            self.assertEqual(handle.readline().strip(), ",".join(HEADER))
        self.assertEqual(read_register(self.path)[1], {"person_id": "P-9002", "preferred_label": "", "notes": "новая"})

    def test_update_merges_given_fields_only(self) -> None:
        result = upsert_rows(self.path, [{"person_id": "P-9001", "notes": "x"}])
        self.assertEqual(result.updated, ("P-9001",))
        self.assertEqual(read_register(self.path)[0], {"person_id": "P-9001", "preferred_label": "Пример", "notes": "x"})

    def test_identical_values_are_reported_unchanged(self) -> None:
        before = digest(self.path)
        result = upsert_rows(self.path, [{"person_id": "P-9001", "preferred_label": "Пример"}])
        self.assertEqual(result.unchanged, ("P-9001",))
        self.assertEqual(digest(self.path), before)

    def test_numbers_are_stored_as_text(self) -> None:
        upsert_rows(self.path, [{"person_id": "P-9002", "notes": 5}])
        self.assertEqual(read_register(self.path)[1]["notes"], "5")

    def test_unknown_column_raises_and_file_untouched(self) -> None:
        before = digest(self.path)
        with self.assertRaises(UnknownColumnsError) as context:
            upsert_rows(self.path, [{"person_id": "P-9002", "bogus": "1"}])
        self.assertIn("bogus", str(context.exception))
        self.assertEqual(digest(self.path), before)

    def test_invalid_input_rows_are_rejected(self) -> None:
        cases = {
            "empty key": [{"person_id": "", "notes": "x"}],
            "whitespace key": [{"person_id": " P-9001 ", "notes": "x"}],
            "duplicate input key": [{"person_id": "P-9002"}, {"person_id": "P-9002", "notes": "y"}],
            "list value": [{"person_id": "P-9002", "notes": [1, 2]}],
            "boolean value": [{"person_id": "P-9002", "notes": True}],
        }
        for label, rows in cases.items():
            with self.subTest(label):
                self.assert_rejected(rows)

    def test_duplicate_ids_in_register_are_rejected(self) -> None:
        write_register(self.path, [{"person_id": "P-9001", "preferred_label": "", "notes": ""}] * 2)
        self.assert_rejected([{"person_id": "P-9001", "notes": "x"}])

    def test_headerless_register_is_rejected(self) -> None:
        self.path.write_text("", encoding="utf-8")
        self.assert_rejected([{"person_id": "P-9001"}])

    def test_bom_register_is_read(self) -> None:
        write_register(self.path, [{"person_id": "P-9001", "preferred_label": "", "notes": ""}], bom=True)
        result = upsert_rows(self.path, [{"person_id": "P-9001", "notes": "x"}])
        self.assertEqual(result.updated, ("P-9001",))

    def test_dry_run_reports_without_writing(self) -> None:
        before = digest(self.path)
        result = upsert_rows(self.path, [{"person_id": "P-9002"}], dry_run=True)
        self.assertEqual(result.added, ("P-9002",))
        self.assertEqual(digest(self.path), before)

    def test_cyrillic_commas_newlines_roundtrip(self) -> None:
        value = "Запись, ѣ і ѳ\nвторая строка"
        upsert_rows(self.path, [{"person_id": "P-9001", "notes": value}])
        self.assertEqual(read_register(self.path)[0]["notes"], value)

    def test_write_leaves_no_temporary_or_lock_files(self) -> None:
        upsert_rows(self.path, [{"person_id": "P-9002"}])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["person-register.csv"])

    def test_existing_lock_blocks_the_write(self) -> None:
        (self.dir / LOCK_NAME).write_text("1", encoding="utf-8")
        self.assert_rejected([{"person_id": "P-9002"}], SourceError)

    def test_register_inside_skill_is_refused(self) -> None:
        with self.assertRaises(SourceError):
            upsert_rows(SKILL_ROOT / "assets" / "person-register.csv", [{"person_id": "P-9001"}])


class StampNowTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        patcher = mock.patch("scripts.upsert_register.timestamp", return_value=STAMP)
        patcher.start()
        self.addCleanup(patcher.stop)

    def register(self, name: str, rows: list[dict[str, str]] = ()) -> Path:
        path = self.dir / name
        write_asset_register(path, rows)
        return path

    def test_stamp_now_fills_missing_time_of_added_rows(self) -> None:
        path = self.register("search-log.csv")
        upsert_rows(path, [{"search_id": "S-9001", "query": "q"}], stamp_now=True)
        self.assertEqual(read_register(path)[0]["timestamp"], STAMP)

    def test_stamp_now_fills_captured_at_in_artifact_register(self) -> None:
        path = self.register("artifact-register.csv")
        upsert_rows(path, [{"artifact_id": "A-9001", "captured_at": ""}], stamp_now=True)
        self.assertEqual(read_register(path)[0]["captured_at"], STAMP)

    def test_stamp_now_notes_registration_time_in_artifact_register(self) -> None:
        path = self.dir / "artifact-register.csv"
        path.write_text("artifact_id,captured_at,reliability_notes\nA-9000,2026-01-01,\n", encoding="utf-8")
        upsert_rows(path, [{"artifact_id": "A-9001", "reliability_notes": "копия"}], stamp_now=True)
        rows = {row["artifact_id"]: row for row in read_register(path)}
        self.assertEqual(rows["A-9001"]["reliability_notes"], "копия Дата исходного получения неизвестна; "
                                                              "captured_at — дата регистрации.")
        self.assertEqual(rows["A-9000"]["reliability_notes"], "")

    def test_stamp_now_through_symlink(self) -> None:
        stored = self.register("search-log.csv").rename(self.dir / "stored.csv")
        link = self.dir / "search-log.csv"
        link.symlink_to(stored)
        upsert_rows(link, [{"search_id": "S-9001"}], stamp_now=True)
        self.assertEqual(read_register(stored)[0]["timestamp"], STAMP)

    def test_stamp_now_finds_time_column_by_header_not_file_name(self) -> None:
        path = self.register("search-log.csv").rename(self.dir / "search-log.2026.csv")
        upsert_rows(path, [{"search_id": "S-9001"}], stamp_now=True)
        self.assertEqual(read_register(path)[0]["timestamp"], STAMP)

    def test_stamp_now_keeps_given_time(self) -> None:
        path = self.register("search-log.csv")
        upsert_rows(path, [{"search_id": "S-9001", "timestamp": "2026-11-01"}], stamp_now=True)
        self.assertEqual(read_register(path)[0]["timestamp"], "2026-11-01")

    def test_stamp_now_leaves_updated_rows_alone(self) -> None:
        path = self.register("search-log.csv", [{"search_id": "S-9001", "timestamp": ""}])
        result = upsert_rows(path, [{"search_id": "S-9001", "next_step": "дальше"}], stamp_now=True)
        self.assertEqual(result.updated, ("S-9001",))
        row = read_register(path)[0]
        self.assertEqual((row["timestamp"], row["next_step"]), ("", "дальше"))

    def test_stamp_now_rerun_keeps_its_own_time(self) -> None:
        path = self.register("artifact-register.csv")
        upsert_rows(path, [{"artifact_id": "A-9001", "captured_at": ""}], stamp_now=True)
        result = upsert_rows(path, [{"artifact_id": "A-9001", "captured_at": ""}], stamp_now=True)
        self.assertEqual(result.unchanged, ("A-9001",))
        self.assertEqual(read_register(path)[0]["captured_at"], STAMP)

    def test_stamp_now_rerun_keeps_the_registration_note(self) -> None:
        path = self.dir / "artifact-register.csv"
        path.write_text("artifact_id,captured_at,reliability_notes\n", encoding="utf-8")
        rows = [{"artifact_id": "A-9001", "captured_at": "", "reliability_notes": "копия из архива"}]
        upsert_rows(path, rows, stamp_now=True)
        result = upsert_rows(path, rows, stamp_now=True)
        self.assertEqual(result.unchanged, ("A-9001",))
        row = read_register(path)[0]
        self.assertEqual(row["captured_at"], STAMP)
        self.assertIn("captured_at — дата регистрации", row["reliability_notes"])

    def test_real_capture_date_drops_the_registration_note(self) -> None:
        path = self.dir / "artifact-register.csv"
        path.write_text("artifact_id,captured_at,reliability_notes\n", encoding="utf-8")
        upsert_rows(path, [{"artifact_id": "A-9001", "reliability_notes": "копия"}], stamp_now=True)
        upsert_rows(path, [{"artifact_id": "A-9001", "captured_at": "1998-05-01"}])
        row = read_register(path)[0]
        self.assertEqual((row["captured_at"], row["reliability_notes"]), ("1998-05-01", "копия"))

    def test_stamp_now_ignored_for_other_registers(self) -> None:
        path = self.dir / "person-register.csv"
        write_register(path, [])
        upsert_rows(path, [{"person_id": "P-9001"}], stamp_now=True)
        self.assertEqual(read_register(path), [{"person_id": "P-9001", "preferred_label": "", "notes": ""}])

    def test_without_stamp_now_time_stays_empty(self) -> None:
        path = self.register("search-log.csv")
        upsert_rows(path, [{"search_id": "S-9001"}])
        self.assertEqual(read_register(path)[0]["timestamp"], "")


class CommandLineTest(unittest.TestCase):
    def run_cli(self, case_dir: Path, register: str, rows: object, *extra: str, raw: str | None = None,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        rows_path = case_dir / "rows.json"
        rows_path.write_text(raw if raw is not None else json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        command = [sys.executable, str(SCRIPTS / "upsert_register.py"), "--case-dir", str(case_dir),
                   "--register", register, "--input", str(rows_path), *extra]
        return subprocess.run(command, capture_output=True, text=True, env=env)

    def make_case(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        case_dir = Path(tmp.name)
        write_register(case_dir / "person-register.csv", [])
        return case_dir

    def test_cli_upserts_and_prints_summary(self) -> None:
        result = self.run_cli(self.make_case(), "person-register.csv", [{"person_id": "P-9001", "notes": "a"}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("added 1, updated 0, unchanged 0", result.stdout)

    def test_cli_dry_run_does_not_write(self) -> None:
        case_dir = self.make_case()
        before = digest(case_dir / "person-register.csv")
        result = self.run_cli(case_dir, "person-register.csv", [{"person_id": "P-9001"}], "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("(dry run)", result.stdout)
        self.assertEqual(digest(case_dir / "person-register.csv"), before)

    def test_cli_stamp_now_sets_time(self) -> None:
        case_dir = self.make_case()
        write_asset_register(case_dir / "search-log.csv")
        result = self.run_cli(case_dir, "search-log.csv", [{"search_id": "S-9001"}], "--stamp-now")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("added 1, updated 0, unchanged 0", result.stdout)
        self.assertRegex(read_register(case_dir / "search-log.csv")[0]["timestamp"],
                         r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")

    def test_cli_reads_json_with_bom(self) -> None:
        raw = "﻿" + json.dumps([{"person_id": "P-9001"}])
        result = self.run_cli(self.make_case(), "person-register.csv", None, raw=raw)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_errors_exit_2_without_traceback(self) -> None:
        cases = {
            "unknown register": ("other.csv", [], None),
            "unknown column": ("person-register.csv", [{"person_id": "P-9001", "bogus": "1"}], None),
            "missing register": ("search-log.csv", [], None),
            "malformed json": ("person-register.csv", None, "[{"),
            "not a list": ("person-register.csv", {"person_id": "P-9001"}, None),
        }
        for label, (register, rows, raw) in cases.items():
            with self.subTest(label):
                result = self.run_cli(self.make_case(), register, rows, raw=raw)
                self.assertEqual(result.returncode, 2)
                self.assertTrue(result.stderr.startswith("Ошибка:"), result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_cli_help_survives_legacy_encoding(self) -> None:
        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        result = subprocess.run([sys.executable, str(SCRIPTS / "upsert_register.py"), "--help"],
                                capture_output=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
