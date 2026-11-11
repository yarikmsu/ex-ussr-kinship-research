"""Воспроизводимый замер privacy_scan: быстрые пути против прямого перебора шаблонов.

- build: поиск слова в тексте навыка через Reference против regex-поиска шаблона по всему тексту;
- check: Matcher (отбор шаблонов по первым трём буквам) против проверки строки каждым шаблоном.

Слова — сгенерированные заглушки из слогов, а не слова дел. Текст навыка и проверяемые
строки — Markdown этой копии навыка. Скрипт сверяет, что оба пути дают одинаковый результат.

Запуск из корня навыка: python3 tests/bench_privacy_scan.py [--words 3000] [--lines 1500]
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from privacy_scan import Matcher, PatternSet, Reference, modern, reference_text, to_pattern  # noqa: E402

SYLLABLES = "ба ве ги до жу зы ко лу ма не по ру са ти фу ха це чу ша ю".split()
ENDINGS = ("ов", "ин", "ский", "енко", "ук", "ова", "ина")


def fake_words(count: int, rng: random.Random) -> list[str]:
    words: set[str] = set()
    while len(words) < count:
        body = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3)))
        words.add(rng.choice("БВГДКЛМНПРСТ") + body + rng.choice(ENDINGS))
    return sorted(words)


def timed(action):
    start = time.perf_counter()
    value = action()
    return time.perf_counter() - start, value


def run(words: int = 3000, lines: int = 1500, seed: int = 1) -> dict[str, float]:
    rng = random.Random(seed)
    names = fake_words(words, rng)
    text = reference_text(ROOT)
    modern_text = modern(text)
    # Часть заглушек вставляется в проверяемые строки, чтобы у обоих путей были совпадения.
    sample = [line for line in text.split("\n") if line.strip()]
    checked = [rng.choice(sample) + (" " + rng.choice(names) if rng.random() < 0.2 else "") for _ in range(lines)]

    build_fast, fast_seen = timed(lambda: lookup_fast(text, names))
    build_regex, regex_seen = timed(lambda: [re.search(to_pattern(name), modern_text, re.IGNORECASE) is not None
                                             for name in names])
    if fast_seen != regex_seen:
        raise AssertionError("Reference и regex по-разному решили, есть ли слово в тексте навыка")

    patterns = PatternSet(tuple(to_pattern(name) for name in names), ())
    lines_modern = [modern(line) for line in checked]
    check_fast, fast_hits = timed(lambda: scan_fast(patterns, lines_modern))
    check_regex, regex_hits = timed(lambda: scan_regex(patterns, lines_modern))
    if sorted(fast_hits) != sorted(regex_hits):
        raise AssertionError("Matcher и прямой перебор нашли разное")
    return {"build_fast": build_fast, "build_regex": build_regex, "check_fast": check_fast, "check_regex": check_regex}


def lookup_fast(text: str, names: list[str]) -> list[bool]:
    reference = Reference(text)  # построение индекса входит в замер
    return [reference.has_word(name) for name in names]


def scan_fast(patterns: PatternSet, lines: list[str]) -> list[tuple[int, str]]:
    matcher = Matcher(patterns)
    return [(hit.line_no, hit.pattern) for number, line in enumerate(lines, 1) for hit in matcher.hits(number, line)]


def scan_regex(patterns: PatternSet, lines: list[str]) -> list[tuple[int, str]]:
    compiled = [(source, re.compile(source, re.IGNORECASE)) for source in patterns.deny]
    return [(number, source) for number, line in enumerate(lines, 1) for source, regex in compiled
            if regex.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--words", type=int, default=3000, help="Число слов-шаблонов")
    parser.add_argument("--lines", type=int, default=1500, help="Число проверяемых строк")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    result = run(args.words, args.lines, args.seed)
    print(f"build: Reference {result['build_fast']:.2f} s, regex {result['build_regex']:.2f} s ({args.words} слов)")
    print(f"check: Matcher {result['check_fast']:.2f} s, перебор {result['check_regex']:.2f} s "
          f"({args.lines} строк × {args.words} шаблонов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
