#!/usr/bin/env python3
"""Добавляет или обновляет строки реестра дела по первому столбцу шапки.

Неизвестный столбец, пустой, повторный или окружённый пробелами ID и
непростое значение — ошибка до записи, а не молчаливая потеря данных.
Обновление меняет только переданные поля. Запись атомарная, под той же
блокировкой, что и у register_search.py, и только вне папки навыка.
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
                             state_lock)

REGISTERS = frozenset({
    "person-register.csv",
    "artifact-register.csv",
    "claims-register.csv",
    "relationship-register.csv",
    "search-log.csv",
})
LOCK_NAME = ".register-search.lock"


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


def save_register(csv_path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(csv_path, stream.getvalue().encode("utf-8"))


def upsert_rows(csv_path: Path, rows: list[dict[str, object]], *, dry_run: bool = False) -> UpsertResult:
    """Сливает строки в реестр по значению первого столбца шапки."""
    csv_path = outside_skill(csv_path)
    if dry_run:
        header, existing = read_register(csv_path)
        added, updated, unchanged = merge(header, existing, normalize_rows(header, rows))
        return UpsertResult(tuple(added), tuple(updated), tuple(unchanged))
    with state_lock(csv_path.parent / LOCK_NAME):
        header, existing = read_register(csv_path)
        added, updated, unchanged = merge(header, existing, normalize_rows(header, rows))
        if added or updated:
            save_register(csv_path, header, existing)
    return UpsertResult(tuple(added), tuple(updated), tuple(unchanged))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Добавляет или обновляет строки реестра дела из JSON.")
    parser.add_argument("--case-dir", required=True, help="Директория дела (вне папки навыка)")
    parser.add_argument("--register", required=True, help="Имя реестра, например person-register.csv")
    parser.add_argument("--input", required=True, help="JSON-массив объектов со столбцами реестра")
    parser.add_argument("--dry-run", action="store_true", help="Показать итог без записи")
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
        result = upsert_rows(Path(args.case_dir).expanduser() / args.register, rows, dry_run=args.dry_run)
    except (OSError, ValueError, csv.Error, SourceError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    suffix = " (dry run)" if args.dry_run else ""
    print(f"added {len(result.added)}, updated {len(result.updated)}, unchanged {len(result.unchanged)}{suffix}")
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
