#!/usr/bin/env python3
"""Добавляет или обновляет строки реестра дела по первому столбцу шапки.

Неизвестный столбец, пустой, повторный или окружённый пробелами ID и
непростое значение — ошибка до записи, а не молчаливая потеря данных.
Обновление меняет только переданные поля; исключение — пометка о дате
регистрации в reliability_notes, которая убирается, когда строке передают
настоящую дату captured_at. Запись атомарная, под той же блокировкой, что
и у register_search.py, и только вне папки навыка.
С --stamp-now пустое поле времени в добавляемых строках журнала поисков
и реестра артефактов заполняется по часам системы в UTC, ISO 8601 — в том
же формате, что у register_search.py. В reliability_notes такой строки
реестра артефактов дописывается, что captured_at — дата регистрации.
С --stamp-now пустое время в обновлении существующей строки её время не стирает.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` omits the script folder

from research_common import (SourceError, atomic_bytes, configure_utf8_stdio, outside_skill, read_json,
                             state_lock, timestamp)

REGISTERS = frozenset({
    "person-register.csv",
    "artifact-register.csv",
    "claims-register.csv",
    "relationship-register.csv",
    "search-log.csv",
})
LOCK_NAME = ".register-search.lock"
# Поле времени по ключевому столбцу шапки: так реестр узнаётся при любом имени файла.
STAMP_COLUMNS: dict[str, str] = {"search_id": "timestamp", "artifact_id": "captured_at"}
# Время из часов — это время регистрации, а не получения материала.
REGISTRATION_NOTE = "Дата исходного получения неизвестна; captured_at — дата регистрации."


class UnknownColumnsError(ValueError):
    """В строке есть ключи, которых нет в шапке реестра."""


@dataclass(frozen=True)
class UpsertResult:
    added: tuple[str, ...]
    updated: tuple[str, ...]
    unchanged: tuple[str, ...]


def read_register(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not header or not header[0]:
        raise ValueError(f"в реестре {csv_path.name} нет шапки")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"повреждена строка CSV в {csv_path.name}")
    ids = [row[header[0]] for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"пустые или повторные ID в {csv_path.name}")
    return header, rows


def normalize_rows(header: list[str], rows: list[dict[str, object]]) -> list[dict[str, str]]:
    """Проверяет входные строки и приводит значения к тексту CSV."""
    key = header[0]
    unknown = sorted({name for row in rows for name in row} - set(header))
    if unknown:
        raise UnknownColumnsError(f"неизвестные столбцы: {', '.join(unknown)}")
    normalized = []
    for row in rows:
        values = {}
        for name, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float, type(None))):
                raise ValueError(f"{name}: ожидается строка или число")
            values[name] = "" if value is None else str(value)
        row_id = values.get(key, "")
        if not row_id:
            raise ValueError(f"строка без значения ключа {key}")
        if row_id != row_id.strip():
            raise ValueError(f"пробелы вокруг ключа {key}: «{row_id}»")
        normalized.append(values)
    input_ids = [values[key] for values in normalized]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError(f"повторный ключ {key} во входных строках")
    return normalized


def merge(header: list[str], existing: list[dict[str, str]],
          rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    key = header[0]
    by_id = {row[key]: row for row in existing}
    added: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    for values in rows:
        row_id = values[key]
        current = by_id.get(row_id)
        if current is None:
            new_row = {name: values.get(name, "") for name in header}
            existing.append(new_row)
            by_id[row_id] = new_row
            added.append(row_id)
        elif all(current.get(name, "") == value for name, value in values.items()):
            unchanged.append(row_id)
        else:
            current.update(values)
            updated.append(row_id)
    return added, updated, unchanged


def drop_stale_notes(header: list[str], existing: list[dict[str, str]],
                     rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Строке реестра артефактов передали настоящую дату captured_at, а reliability_notes нет: пометка,
    что captured_at — дата регистрации, больше не верна и убирается из старого текста."""
    if header[0] != "artifact_id" or "reliability_notes" not in header:
        return rows
    by_id = {row[header[0]]: row for row in existing}
    cleaned = []
    for values in rows:
        current = by_id.get(values[header[0]])
        date = values.get("captured_at", "")
        if (current is not None and date and date != current.get("captured_at") and "reliability_notes" not in values
                and REGISTRATION_NOTE in current.get("reliability_notes", "")):
            notes = current["reliability_notes"].replace(REGISTRATION_NOTE, "")
            values = {**values, "reliability_notes": " ".join(notes.split())}
        cleaned.append(values)
    return cleaned


def keep_stamps(header: list[str], existing: list[dict[str, str]],
                rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """С --stamp-now пустое время значит «поставь время», а не «сотри»: у существующей строки его не трогают.

    Если время строки осталось от --stamp-now, пометка о дате регистрации в reliability_notes тоже
    остаётся: иначе повтор того же JSON выдал бы время регистрации за дату получения.
    """
    column = STAMP_COLUMNS.get(header[0])
    by_id = {row[header[0]]: row for row in existing}
    kept = []
    for values in rows:
        values = dict(values)
        current = by_id.get(values[header[0]])
        if current is not None and column:
            if values.get(column) == "":
                del values[column]
            notes = values.get("reliability_notes")
            if (column == "captured_at" and column not in values and notes is not None
                    and REGISTRATION_NOTE in current.get("reliability_notes", "") and REGISTRATION_NOTE not in notes):
                values["reliability_notes"] = " ".join(filter(None, [notes, REGISTRATION_NOTE]))
        kept.append(values)
    return kept


def stamp_added(header: list[str], rows: list[dict[str, str]], added: list[str]) -> None:
    """Ставит текущее время в пустое поле времени добавленных строк."""
    column = STAMP_COLUMNS.get(header[0])
    if column not in header:
        return
    added_ids = set(added)
    now = timestamp()
    for row in rows:
        if row[header[0]] in added_ids and not row[column]:
            row[column] = now
            if column == "captured_at" and "reliability_notes" in row:  # как в register_search.py
                row["reliability_notes"] = " ".join(filter(None, [row["reliability_notes"], REGISTRATION_NOTE]))


def save_register(csv_path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(csv_path, stream.getvalue().encode("utf-8"))


def merge_into(csv_path: Path, rows: list[dict[str, object]],
               stamp_now: bool) -> tuple[list[str], list[dict[str, str]], UpsertResult]:
    header, existing = read_register(csv_path)
    normalized = drop_stale_notes(header, existing, normalize_rows(header, rows))
    if stamp_now:
        normalized = keep_stamps(header, existing, normalized)
    added, updated, unchanged = merge(header, existing, normalized)
    if stamp_now:
        stamp_added(header, existing, added)
    return header, existing, UpsertResult(tuple(added), tuple(updated), tuple(unchanged))


def upsert_rows(csv_path: Path, rows: list[dict[str, object]], *, dry_run: bool = False,
                stamp_now: bool = False) -> UpsertResult:
    """Сливает строки в реестр по значению первого столбца шапки."""
    csv_path = outside_skill(csv_path)
    if dry_run:
        return merge_into(csv_path, rows, stamp_now)[2]
    with state_lock(csv_path.parent / LOCK_NAME):
        header, existing, result = merge_into(csv_path, rows, stamp_now)
        if result.added or result.updated:
            save_register(csv_path, header, existing)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Добавляет или обновляет строки реестра дела из JSON.")
    parser.add_argument("--case-dir", required=True, help="Директория дела (вне папки навыка)")
    parser.add_argument("--register", required=True, help="Имя реестра, например person-register.csv")
    parser.add_argument("--input", required=True, help="JSON-массив объектов со столбцами реестра")
    parser.add_argument("--dry-run", action="store_true", help="Показать итог без записи")
    parser.add_argument("--stamp-now", action="store_true",
                        help="Заполнить пустое время добавляемых строк (timestamp, captured_at) "
                             "по часам системы: UTC, ISO 8601; для captured_at дописать в reliability_notes, "
                             "что это дата регистрации")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.register not in REGISTERS:
        print(f"Ошибка: неизвестный реестр {args.register}; допустимы: {', '.join(sorted(REGISTERS))}",
              file=sys.stderr)
        return 2
    try:
        rows = read_json(args.input)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("ожидается JSON-массив объектов")
        result = upsert_rows(Path(args.case_dir).expanduser() / args.register, rows, dry_run=args.dry_run,
                             stamp_now=args.stamp_now)
    except (OSError, ValueError, csv.Error, SourceError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    suffix = " (dry run)" if args.dry_run else ""
    print(f"added {len(result.added)}, updated {len(result.updated)}, unchanged {len(result.unchanged)}{suffix}")
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
