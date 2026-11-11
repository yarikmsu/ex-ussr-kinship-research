#!/usr/bin/env python3
"""Проверяет исходящий текст на имена, места и ID из реестров дела.

build собирает из реестров дела регулярные выражения. Слова дела идут в
список LEAK, а слова, которые уже есть в документах навыка, — в список
REVIEW. Слово, которое документы навыка пишут со строчной или заглавными
(«ЦАМО»), считается обычным словом или сокращением и шаблона не получает;
все отсеянные так слова полей имён и мест build называет в предупреждении.
Слово свободного текста, которое реестры пишут и со строчной, идёт в
REVIEW; в полях имён и мест строчное написание в самих реестрах его не
ослабляет: фамилия «Мельник» остаётся в LEAK. Дела собираются по
отдельности: allow дела действует только на его слова. Слова LEAK ищутся
и в латинском написании. Документы навыка в рабочей копии git — только
отслеживаемые файлы. Частые имена узнаются по падежным формам; мужская
форма отчества идёт в REVIEW: она бывает фамилией.
check проверяет текст или diff по этим спискам: LEAK даёт код выхода 1,
REVIEW только печатается. Файл шаблонов содержит данные дела, поэтому
он пишется только вне папки навыка и вне --reference-dir. Частные
дополнения дела (редкие имена предков, ключевые топонимы, исключения)
хранятся в <дело>/privacy-scan.json. Замер скорости — tests/bench_privacy_scan.py.

Слова короче четырёх букв и латиница из реестров не берутся: такие имена
указывают в extra_deny. Текст и шаблоны сравниваются в современной
орфографии: ѣ, і, ѳ, ѵ заменяются на е, и, ф, и; знаки ударения и мягкий
перенос удаляются.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` omits the script folder

from research_common import SourceError, configure_utf8_stdio, outside_skill, read_json, save_json
from upsert_register import REGISTERS

CONFIG_NAME = "privacy-scan.json"
CONFIG_KEYS = frozenset({"extra_deny", "allow"})
# Текст навыка — документы. Код и тесты не берутся: их примеры (вымышленные фамилии
# тестов) перевели бы настоящие слова дела из LEAK в REVIEW.
REFERENCE_SUFFIXES = (".md", ".yaml")
REFERENCE_SKIPPED_DIRS = frozenset({".git", "tests"})
# Для ID — весь текст навыка, с кодом и тестами: нумерация ID у тестов та же, что у дел.
ID_SOURCE_SUFFIXES = (".md", ".yaml", ".py", ".csv", ".json", ".txt", ".ged")
# Поисковые строки журнала пишут со строчной и имена: их строчные слова не делают имя обычным словом.
QUERY_FIELDS = frozenset({"query", "filters"})
# Папки навыка, которые читаются из копии без git: заметки дел в других папках в эталон не попадают.
SKILL_FOLDERS = ("references", "evals", "assets", "scripts", "tests")
# Поля с именами людей: каждое слово с заглавной — кандидат, даже в начале поля и из одних заглавных.
NAME_FIELDS = frozenset({"names_as_written", "preferred_label", "normalized_name_variants"})
# Поля с местами: то же. Форма частого имени («Николы», «Никола») и здесь не кандидат:
# в эти поля попадают и имена людей. Такое название указывают в extra_deny.
PLACE_FIELDS = frozenset({"places", "place_as_written"})
# Поля-перечни: правила свободного текста, но первое слово поля — тоже кандидат («Гончарова Мария (мать)»).
LIST_FIELDS = frozenset({"relatives_as_reported", "historical_jurisdictions", "query"})
UPPER = "А-ЯЁІЇЄҐЎѲѢѴ"
LOWER = "а-яёіїєґўѳѣѵ"
LETTER = UPPER + LOWER
APOSTROPHES = "'’ʼ"
MIN_LETTERS = 4
MIN_STEM = 5
_TAIL = rf"(?:[{LETTER}]|[{APOSTROPHES}](?=[{LETTER}]))*"
WORD = re.compile(rf"(?<![{LETTER}{APOSTROPHES}])[{UPPER}]{_TAIL}(?:-[{LETTER}]{_TAIL})*")
CAPS_NEXT = re.compile(rf"\s+([{UPPER}]\.|[{UPPER}][{LOWER}]+)")
LOWER_WORD = re.compile(rf"(?<![{LETTER}])[{LOWER}][{LOWER}-]{{3,}}")
LETTER_RUN = re.compile(f"[{LETTER}]+")
HAS_LETTER = re.compile(f"[{LETTER}]")
ID_TOKEN = re.compile(r"[\w-]+")
ID_SHAPE = re.compile(r"([A-Za-z]+)-(\d+)")
NEXT_IDS = 50
# Латинское написание: одна транслитерация кириллицы, затем общий «скелет» для разных систем.
TRANSLIT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i", "й": "y",
            "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
            "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e",
            "ю": "yu", "я": "ya", "і": "i", "ї": "i", "є": "ye", "ґ": "g", "ў": "u"}
LATIN_DIGRAPHS = {"shch": "s", "sch": "s", "zh": "z", "kh": "h", "ch": "c", "sh": "s", "ts": "c", "tz": "c",
                  "cz": "c", "sz": "s", "ya": "a", "ja": "a", "yu": "u", "ju": "u", "yo": "e", "jo": "e",
                  "ye": "e", "je": "e"}
LATIN_DIGRAPH = re.compile("|".join(sorted(LATIN_DIGRAPHS, key=len, reverse=True)))
LATIN_LETTERS = str.maketrans("yjwq", "iivk", "'’ʼ")
_LATIN = "A-Za-zÀ-ÖØ-öø-ɏ"  # латиница, в том числе с диакритикой
LATIN_TOKEN = re.compile(rf"(?<![{_LATIN}])[{_LATIN}]+(?:['’ʼ][{_LATIN}]+)?")
MIN_LATIN_KEY = 5
# Запись extra_deny из букв не кириллицы (латиница, в том числе с диакритикой): ищется по началу слова.
LATIN_WORD = re.compile(rf"[^\W\d_](?:(?:[^\W\d_]|[{APOSTROPHES}-])*[^\W\d_])?")
# Номера из полей-ссылок реестров: шифр с буквой и цифрой через дефис («Р-1907», ark FamilySearch) и хеши.
REFERENCE_FIELDS = frozenset({"archival_reference", "source_url_or_path", "sha256"})
REFERENCE_NUMBER = re.compile(r"(?<![\w-])(?=[\w-]*\d)(?=[\w-]*[^\W\d_])\w+(?:-\w+)+(?![\w-])"
                              r"|(?<![0-9A-Fa-f])[0-9A-Fa-f]{32,}(?![0-9A-Fa-f])")
ORDINAL = re.compile(rf"\d+-[{LOWER}]{{1,3}}")  # «125-й», «1905-го» — порядковые числа, а не шифры
MIN_REFERENCE_NUMBER = 3
VOWELS = "аеёиоуыэюяіїєў"
SIBILANTS = "жшщчц"  # после них творительный падеж на -ей: «Квашей»
# Короткая притяжательная фамилия («Шуева», «Ивиной»): основа с суффиксом не короче четырёх букв.
POSSESSIVE = re.compile(rf"([{LETTER}]{{2,}}(?:ов|ев|ёв|ин|ын))(?:а|ой|ою|у|ы|ым|ых|ыми|е|ом)?")
HUNK = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
# Старые буквы → современные; ѐ и ѝ — буквы с ударением после NFC; варианты начертания кириллицы
# (U+1C80–U+1C88, «ᲀ» — круглое «в») → обычные буквы; апострофы ’ и ʼ → '; ударение и мягкий перенос удаляются.
OLD_LETTERS = str.maketrans("ѣѢіІѳѲѵѴѐЀѝЍ\u1c80\u1c81\u1c82\u1c83\u1c84\u1c85\u1c86\u1c87\u1c88\u2019\u02bc",
                            "еЕиИфФиИеЕиИвдосттъеу''", "\u0300\u0301\u00ad")
# Старое «-ския», «-цкия» («Белевскія») — в современное «-ские»: в современном письме слова так не
# кончаются, а «-ия» в шаблонах ловило бы «Швеция» по «Швец» и «Бессарабия» по «Бессараб».
OLD_SKIYA = re.compile(rf"(?<=[сцСЦ][кК])[иИ][яЯ](?![{LETTER}])")
# Невидимые знаки форматирования Unicode (категория Cf): мягкий перенос, пробел нулевой ширины и т. п.
INVISIBLE = re.compile("[" + "".join(chr(code) for code in range(0x110000)
                                     if unicodedata.category(chr(code)) == "Cf")
                       + "\u034f\ufe00-\ufe0f\u180b-\u180d\U000e0100-\U000e01ef]")
# Латинские буквы, похожие на кириллические; заменяются только в словах, где есть кириллица.
LOOKALIKE_LATIN = "aceopxyiABCEHKMOPTXI"
LOOKALIKE_CYRILLIC = "асеорхуіАВСЕНКМОРТХІ"
LOOKALIKES = str.maketrans(LOOKALIKE_LATIN, LOOKALIKE_CYRILLIC)
ASCII_LETTER = re.compile("[A-Za-z]")
MIXED_WORD = re.compile(r"[^\W\d_]+")
# Окончания рода и падежа, которые отрезаются перед построением шаблона, длинные первыми.
# «-аго», «-яго», «-ыя» — окончания прилагательных старой орфографии («Залесскаго»); у короткой
# основы они не отрезаются, и застывшая фамилия «Живаго» остаётся словом целиком.
STEM_ENDINGS = ("ыми", "ими", "ого", "его", "ому", "ему", "аго", "яго", "ых", "их", "ым", "им", "ой", "ей",
                "ою", "ею", "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие", "ыя", "ий", "ый", "ом", "ем", "ам",
                "ям", "ах", "ях", "а", "я", "е", "у", "ю", "ы", "и", "о")
# Окончания старой орфографии: отрезаются только у длинной основы и не дают латинских ключей.
OLD_ENDINGS = ("аго", "яго", "ыя")
# Окончания, с которыми шаблон находит основу.
PATTERN_ENDINGS = STEM_ENDINGS
# Окончания прилагательного: короткая фамилия «Белая», «Седой» ищется по основе «Бел», «Сед» с ними.
# Старые окончания основу прилагательного не дают («Живаго» — не «Жив»), но шаблон их находит («Белаго»).
ADJ_ENDINGS = ("ыми", "ими", "ого", "его", "ому", "ему", "ых", "их", "ым", "им", "ой", "ей", "ою", "ею",
               "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие", "ий", "ый", "ом", "ем")
ADJ_PATTERN_ENDINGS = ADJ_ENDINGS + OLD_ENDINGS
MIN_ADJ_STEM = 3
# С какими окончаниями шаблон находит основу: любое из PATTERN_ENDINGS или никакого,
# а за ним может стоять старый «ъ»: «Тихомировъ», «Тихомировымъ».
WORD_ENDINGS = tuple(end + hard for end in ("",) + PATTERN_ENDINGS for hard in ("", "ъ"))
# Сокращения реестров, после которых слово с заглавной — середина фразы. Сокращения
# короче четырёх букв («д.», «с.», «губ.») узнаются по длине.
ABBREVIATIONS = frozenset("""
урожд урож крест крестьян крещ свящ мещ мещан солд рядов отст волост уезд прих прихож церк губерн станиц
респ колх совх прож посел слоб погр венч вдов однодв дворов помещ казен удельн войск окруж сельсов
горсов райисп облисп красноарм лейт капит серж ефрейт старш младш учит служ рабоч колхозн
""".split())
# Служебные слова — закрытый класс, фамилией не бывают: «Того же села» в поле имён.
FUNCTION_WORDS = frozenset("""
того тому тех теми этот этого этому этим этом этой эти этих этими сего сему сии оного оному оной оных
оный оная оное него нему ними всех всем всего всему весь всеми сама само сами самого самой самих самим
свой своя свое свои своего своей своих своим моего моей моих наша наше наши нашего нашей наших ваша
ваше ваши вашего вашей ваших также тоже чтобы когда если потом затем после перед между около через
возле вместо кроме ради тогда здесь туда сюда оттуда откуда куда кого кому чего чему какой какая какое
какие каких который которая которое которые которого которой которых которым каждый каждая каждое
каждого каждой другой другая другое другие другого других один одна одно одни одного одной одних обоих
обеих только даже будто либо нибудь поэтому потому однако хотя пока вдоль внутри вокруг сверх среди
против ввиду вроде насчет сквозь
""".split())
# Частые имена XIX–XX веков — общая лексика, а не данные дела. Каждое подтверждено открытым
# источником, не делом: глоссарием частых имён BYU Script Tutorial (script.byu.edu, Russian
# handwriting: masculine и feminine names), текстом основного навыка или как церковная или
# бытовая пара такого имени (Аксинья — Ксения). Список неполный: частое имя, которого здесь
# нет, дело вписывает в allow. Имя, которое для дела редкое, указывают в extra_deny.
GIVEN_NAMES = frozenset("""
Авдотья Аграфена Аксинья Акулина Александр Александра Алексей Алена Алла Алёна Анастасия Андрей
Анисья Анна Антон Антонина Арина Афанасий Борис Вадим Валентин Валентина Варвара Василий Вера Виктор
Владимир Владислав Вячеслав Гавриил Гаврила Галина Ганна Георгий Герасим Герман Григорий Даниил
Данила Дарья Денис Димитрий Дмитрий Евгений Евгения Евдокия Егор Екатерина Елена Елизавета Емельян
Ефим Зинаида Зоя Иаков Иван Игнат Игнатий Игорь Илия Илья Иоанн Иосиф Ирина Кирилл Кондрат
Константин Ксения Лев Леонид Лидия Любовь Людмила Макар Максим Маргарита Марина Мария Мартын Матрена
Матрёна Михаил Михайла Мыкола Надежда Наталия Наталья Никита Никифор Никола Николай Нина Оксана Олег
Ольга Осип Павел Пелагея Петр Петро Платон Прасковья Пётр Раиса Роман Светлана Семен Семён Сергей
Симеон София Софья Станислав Степан Стефан Тамара Татьяна Тимофей Ульяна Устин Устинья Федор Феодор
Филипп Фома Фёдор Христина Эдуард Юрий Яков
""".split())


@dataclass(frozen=True)
class PatternSet:
    deny: tuple[str, ...]
    review: tuple[str, ...]
    notes: tuple[str, ...] = ()  # предупреждения build; в файл шаблонов не пишутся
    latin: tuple[str, ...] = ()  # ключи латинского написания слов LEAK


@dataclass(frozen=True)
class Hit:
    kind: Literal["LEAK", "REVIEW"]
    line_no: int
    pattern: str
    text: str
    fragment: str = ""


def modern(text: str) -> str:
    """Составные буквы в NFC, современные буквы вместо ѣ, і, ѳ, ѵ, без ударений и невидимых знаков.

    Невидимые знаки форматирования (мягкий перенос, пробел нулевой ширины) удаляются, а латинские
    буквы-двойники внутри кириллического слова («Тихомирoв» с латинской o) заменяются кириллицей.
    """
    text = unicodedata.normalize("NFC", INVISIBLE.sub("", text))
    text = MIXED_WORD.sub(lookalikes_to_cyrillic, text).translate(OLD_LETTERS)
    return OLD_SKIYA.sub(lambda match: "ие" if match.group(0).islower() else "ИЕ", text)


def lookalikes_to_cyrillic(match: re.Match[str]) -> str:
    """Кириллическое слово с латинскими двойниками («Тихомирoв») — в кириллицу; «Kowalczykом» не трогается."""
    word = match.group(0)
    if HAS_LETTER.search(word) and all(char in LOOKALIKE_LATIN for char in ASCII_LETTER.findall(word)):
        return word.translate(LOOKALIKES)
    return word


def fold(text: str) -> str:
    return text.lower().replace("ё", "е")


def name_form(word: str) -> str:
    """Слово реестра в обычном написании: без конечного ъ, заглавная только первая буква."""
    parts = [part[:-1] if part.endswith(("ъ", "Ъ")) else part for part in modern(word).split("-")]
    if all(part.isupper() for part in parts if part):
        parts = [part[:1] + part[1:].lower() for part in parts]
    return "-".join(parts)


# Мужские имена на -а/-я из списка частых имён. От одних отчество на -ич («Ильич», «Никитична»),
# от других на -ович («Данилович»). От женских имён отчеств нет.
MALE_A_NAMES = frozenset({"Илья", "Никита", "Фома"})
MALE_A_OVICH_NAMES = frozenset({"Данила", "Гаврила", "Михайла"})
# Отчество от церковной или местной формы имени — как от обычной: «Илия» → «Ильич», «Петро» → «Петрович».
PATRONYMIC_NAME = {"Илия": "Илья", "Никола": "Николай", "Мыкола": "Николай", "Петро": "Пётр"}
PATRONYMIC_ENDINGS = ("", "а", "у", "ем", "е", "ы", "ой", "ою")


def patronymic_bases(name: str) -> set[str]:
    """Основы отчеств от имени в свёртке: «Иван» → «иванович», «ивановн»; «Юрий» → «юрьевич», «юрьевн»."""
    name = PATRONYMIC_NAME.get(name, name)
    folded = fold(name)
    if name in MALE_A_OVICH_NAMES:
        stems = {folded[:-1] + "ов"}
    elif folded.endswith(("а", "я")):
        root = folded[:-1]
        return {root + "ич", root + "ичн", root + "иничн"} if name in MALE_A_NAMES else set()
    elif folded in ("павел", "лев"):  # беглое «е»
        stems = {folded[:-2] + ("ьв" if folded == "лев" else folded[-1]) + "ов"}
    elif folded.endswith("аил"):  # «аил» → «айл» перед -ович
        stems = {folded[:-2] + "йлов"}
    elif folded.endswith(("яков", "иаков")):  # -ов → -овлев, в том числе у церковной формы
        stems = {folded + "лев"}
    elif folded.endswith("ий"):
        stems = {folded[:-2] + "ьев", folded[:-1] + "ев"}
    elif folded.endswith(("й", "ь")):
        stems = {folded[:-1] + "ев"}
    else:
        stems = {folded + "ов"}
    return {stem_ + "ич" for stem_ in stems} | {stem_ + "н" for stem_ in stems}


PATRONYMIC_BASES = frozenset(base for name in GIVEN_NAMES for base in patronymic_bases(name))


def name_forms(name: str) -> set[str]:
    """Падежные формы частого имени в свёртке: «Андрей» → «андрея», «андрею»…; «Андр» среди них нет."""
    folded = fold(name)
    if folded in ("павел", "лев"):  # беглое «е»
        base, ends = folded[:-2] + ("ьв" if folded == "лев" else folded[-1]), ("а", "у", "ом", "е")
    elif folded.endswith("ий"):
        base, ends = folded[:-1], ("я", "ю", "ем", "и")
    elif folded.endswith(("ей", "ай", "ой")):
        base, ends = folded[:-1], ("я", "ю", "ем", "е")
    elif folded.endswith("ь"):
        base, ends = folded[:-1], ("я", "ю", "ем", "е", "и", "ью")
    elif folded.endswith("ия"):
        base, ends = folded[:-1], ("и", "ю", "ей", "ею")
    elif folded.endswith("я"):
        base, ends = folded[:-1], ("и", "е", "ю", "ей", "ею")
    elif folded.endswith("а"):
        base, ends = folded[:-1], ("ы", "и", "е", "у", "ой", "ою")
    elif folded.endswith("о"):
        base, ends = folded[:-1], ("а", "у", "ом", "е")
    else:
        base, ends = folded, ("а", "у", "ом", "ем", "е")
    return {folded} | {base + end for end in ends}


GIVEN_FORMS = frozenset(form for name in GIVEN_NAMES for form in name_forms(name))


def given_name_kind(word: str) -> str | None:
    """«form» — падежная форма частого имени; «fragment» — обрывок, шаблон которого поймал бы имя («Димитр»)."""
    if fold(word) in GIVEN_FORMS:
        return "form"
    body = fold(stem(word))
    return "fragment" if any(body + end in GIVEN_FORMS for end in WORD_ENDINGS) else None


def is_given_name(word: str) -> bool:
    return given_name_kind(word) is not None


def is_name_abbreviation(word: str, following: str) -> bool:
    """Сокращение частого имени с точкой: «Конст.». Без точки решает given_name_kind: «Евген» — обрывок
    имени (шаблона нет, build называет его в предупреждении), «Констан» — слово-кандидат."""
    return following.startswith(".") and any(name.startswith(word) for name in GIVEN_NAMES)


def patronymic_kind(word: str) -> str | None:
    """Отчество от целого частого имени в любом падеже: «male» («Иванович», «Ильич») или «female» («Юрьевна»).

    «Андрич», «Борович», «Рабинович» и «Бабич» — фамилии: их основа не равна имени.
    """
    folded = fold(word)
    base = next((folded[: len(folded) - len(end)] for end in PATRONYMIC_ENDINGS
                 if folded.endswith(end) and folded[: len(folded) - len(end)] in PATRONYMIC_BASES), None)
    return None if base is None else "male" if base.endswith("ич") else "female"


def is_patronymic(word: str) -> bool:
    return patronymic_kind(word) is not None


def stem(word: str) -> str:
    """Отрезает самое длинное подходящее окончание, если основа не короче MIN_STEM.

    Короткую основу не продлевают более коротким окончанием: «Старое» не
    становится «Старо», иначе шаблон ловил бы «по старому стилю».
    """
    end = next((end for end in STEM_ENDINGS if word.endswith(end)), "")
    if end and len(word) - len(end) >= MIN_STEM:
        return word[: -len(end)]
    possessive = POSSESSIVE.fullmatch(word)  # «Шуева» → «Шуев»: короткая фамилия на -ов, -ин
    return possessive.group(1) if possessive else word


def word_key(word: str) -> str:
    return fold(stem(word))


def lexeme_key(word: str) -> str:
    """Общий ключ форм слова для allow: «Ивина», «Ивиной», «Ивиным» → «ивин»; «Тихомировым» → «тихомиров».

    Короткой основе (меньше MIN_STEM) окончание отрезается до трёх букв основы.
    """
    adjective = adjective_stem(word)
    if adjective:
        return fold(adjective)
    base = stem(word)
    if base == word:
        # Старое «-аго» короткой основы не отрезается: «Живаго» — не форма «Живой».
        end = next((end for end in STEM_ENDINGS if end not in OLD_ENDINGS and word.endswith(end)
                    and len(word) - len(end) >= 3), "")
        base = word[: len(word) - len(end)]
    return fold(base)


def adjective_stem(word: str) -> str | None:
    """Основа короткой фамилии-прилагательного («Белая» → «Бел»), если обычная основа была бы короче MIN_STEM."""
    end = next((end for end in ADJ_ENDINGS if word.endswith(end)), "")
    rest = len(word) - len(end)
    if (not end or not MIN_ADJ_STEM <= rest < MIN_STEM or POSSESSIVE.fullmatch(word)
            or any(fold(word[:rest]) + adj in GIVEN_FORMS for adj in ADJ_ENDINGS)):
        # «Фомий»: основа «Фом» ловила бы «Фомой» (форму имени «Фома»); «Шуевой» — притяжательная фамилия.
        # «Тихая» основу получает: частое имя лишь начинается так же.
        return None
    return word[:rest]


def yo_free(text: str) -> str:
    """Регулярное выражение для текста, где е и ё взаимозаменяемы посимвольно."""
    return "".join("[её]" if char.lower() in "её" else re.escape(char) for char in text)


ENDINGS_PATTERN = "|".join(yo_free(end) for end in PATTERN_ENDINGS)
ADJ_ENDINGS_PATTERN = "|".join(yo_free(end) for end in ADJ_PATTERN_ENDINGS)


def to_pattern(word: str) -> str:
    """Шаблон целого слова: основа и одно из окончаний WORD_ENDINGS.

    Без конца слова короткая основа ловила бы обычные слова: «Добров» — «добровольно».
    """
    return f"(^|[^{LETTER}]){yo_free(stem(word))}(?:{ENDINGS_PATTERN})?ъ?(?![{LETTER}])"


def ending_pattern(stem_: str) -> str:
    """Основа и обязательное окончание: косвенные формы с беглой гласной («Травц» → «Травца»)."""
    return f"(^|[^{LETTER}]){yo_free(stem_)}(?:{ENDINGS_PATTERN})ъ?(?![{LETTER}])"


def exact_pattern(word: str) -> str:
    """Только эта форма слова: соединительная часть топонима («Хмурово-») или именительный «Травец»."""
    return f"(^|[^{LETTER}]){yo_free(word)}ъ?(?![{LETTER}])"


def fleeting_patterns(word: str) -> list[str]:
    """Формы с беглой гласной: «Травец» — «Травца», «Хорел» — «Хорла», «Буровей» — «Буровья»; «Травца» — «Травец»."""
    lower = fold(word)  # «Орёл» — как «Орел»: «Орла»
    if len(word) >= 4 and lower[-3] not in VOWELS and lower.endswith(("ец", "ок", "ек", "ел")):
        return [ending_pattern(word[:-2] + word[-1])]
    if len(word) >= 5 and lower[-3] not in VOWELS and lower.endswith("ей"):
        return [ending_pattern(word[:-2] + "ь")]
    base = stem(word)
    if base != word and base.lower().endswith("ц") and base.lower()[-2] not in VOWELS:
        return [exact_pattern(base[:-1] + "ец")]
    return []


def adjective_pattern(stem_: str) -> str:
    """Шаблон короткой фамилии-прилагательного: основа и обязательное окончание прилагательного."""
    return f"(^|[^{LETTER}]){yo_free(stem_)}(?:{ADJ_ENDINGS_PATTERN})ъ?(?![{LETTER}])"


def is_reference_code(token: str) -> bool:
    """Шифр, а не слово с номером: «Р-1907», «3QS7-L9QQ», но не «125-й» и не «image-001»."""
    if len(token) < MIN_REFERENCE_NUMBER or ORDINAL.fullmatch(token):
        return False
    if "-" not in token:
        return True  # хеш
    groups = token.split("-")
    mixed = any(any(char.isdigit() for char in group) and any(char.isalpha() for char in group) for group in groups)
    return mixed or all(len(group) <= 3 for group in groups if group.isalpha())


def code_pattern(value: str) -> str:
    """Шифр с буквами-двойниками любой письменности: «Р-12» кириллицей ловит и «P-12» латиницей.

    Хеш и длинный латинский код (ark FamilySearch) ищутся как есть: их пишут только латиницей,
    а классы букв в сотнях таких шаблонов замедляли бы проверку.
    """
    if not HAS_LETTER.search(value) and len(value) > 8:
        return id_pattern(value)
    pairs = dict(zip(LOOKALIKE_LATIN + LOOKALIKE_CYRILLIC, LOOKALIKE_CYRILLIC + LOOKALIKE_LATIN))
    body = "".join(f"[{re.escape(char)}{re.escape(pairs[char])}]" if char in pairs else re.escape(char)
                   for char in value)
    return rf"(?<![\w-]){body}(?![\w-])"


def id_pattern(value: str) -> str:
    return rf"(?<![\w-]){re.escape(value)}(?![\w-])"


def latin_pattern(word: str) -> str:
    """Латинское имя: начало слова и любое продолжение — «Tikhomirov» ловит «Tikhomirova», «Tikhomirov's»."""
    return rf"(?<![\w-]){re.escape(word)}(?:[^\W\d_]|[{APOSTROPHES}])*"


def next_ids(ids: Iterable[str]) -> set[str]:
    """Следующие NEXT_IDS номеров после последнего ID каждой серии: «P-9042» → «P-9043» … «P-9092».

    Такие ID пишут в план и передачу раньше, чем появятся строки реестра.
    """
    last: dict[tuple[str, int], int] = {}
    for value in ids:
        if match := ID_SHAPE.fullmatch(value):
            series = (match.group(1), len(match.group(2)))
            last[series] = max(last.get(series, 0), int(match.group(2)))
    return {f"{prefix}-{number:0{width}d}" for (prefix, width), top in last.items()
            for number in range(top + 1, top + 1 + NEXT_IDS) if len(str(number)) <= width}


def is_code_entry(entry: str) -> bool:
    """Запись extra_deny — шифр («Р-1907»), а не слово: ищется как шифр реестров."""
    return REFERENCE_NUMBER.fullmatch(entry) is not None and is_reference_code(entry)


def entry_pattern(entry: str) -> str:
    """Запись extra_deny: кириллица — все формы слова, шифр — с буквами-двойниками, латинское имя — начало слова,
    остальное — точный токен."""
    entry = modern(entry.strip())
    if is_code_entry(entry):
        return code_pattern(entry)
    if HAS_LETTER.search(entry):
        word = name_form(entry)
        adjective = adjective_stem(word)
        return adjective_pattern(adjective) if adjective else to_pattern(word)
    return latin_pattern(entry) if LATIN_WORD.fullmatch(entry) else id_pattern(entry)


def entry_patterns(entry: str) -> list[str]:
    """Все шаблоны записи extra_deny: основной, формы с беглой гласной и падежные формы.

    Имя из списка частых имён ищется точными формами единственного и множественного числа:
    короткую основу «Фом» шаблон слова не отрезает, а основа «Лев» ловила бы «левый», «Виктор» —
    «Виктория», «Александра» — «Александр».
    """
    word = name_form(modern(entry.strip()))
    name = listed_name(entry)
    if name:  # только точные формы: у «Лев» шаблон основы ловил бы «левый», у «Виктор» — «Виктория»
        return [exact_pattern(form.capitalize()) for form in sorted(name_forms(name) | plural_forms(name))]
    if is_code_entry(modern(entry.strip())) or not HAS_LETTER.search(word):
        return [entry_pattern(entry)]
    return [entry_pattern(entry)] + fleeting_patterns(word) + form_patterns(word)


@lru_cache(maxsize=None)
def declined_forms(word: str) -> frozenset[str]:
    """Падежные формы существительного в свёртке: «Сызрань» → «сызрани», «Поречие» → «поречия», «Кваша» → «квашей»."""
    folded = fold(word)
    if folded.endswith("ие"):
        return frozenset({folded} | {folded[:-1] + end for end in ("я", "ю", "ем", "и")})
    forms = name_forms(word)
    if folded.endswith("а") and len(folded) > 1 and folded[-2] in SIBILANTS:
        forms |= {folded[:-1] + "ей", folded[:-1] + "ею"}
    if folded.endswith(("ней", "жей", "шей", "чей", "щей")):  # «Зимней» → не «зимнее»: это обычное слово
        forms.discard(folded[:-2] + "ее")
    # После г, к, х и шипящих пишется «и», а не «ы»: «Кваши», а не «Квашы».
    return frozenset(re.sub("(?<=[гкхжшщч])ы", "и", form) for form in forms)


def needs_forms(word: str) -> bool:
    """Слово, падежные формы которого шаблон основы не находит.

    У «Анатолий» и «Поречие» основа косвенных форм другая («Анатоли-»), у «Коваль», «Сызрань»,
    «Ерофей» и «Бугай» окончание мягкое, а короткую основу («Кваш», «Агни») stem не отрезает.
    Прилагательные («-ский», «-ская») и косвенные формы («Мариных») шаблон основы находит сам.
    """
    folded = fold(word)
    if folded.endswith(("ский", "цкий", "ские", "цкие")):
        return False
    if folded.endswith(("ий", "ие", "ь", "ай", "ей")):
        return True
    return folded.endswith(("а", "я", "о")) and stem(word) == word


def missing_forms(word: str, *, noun: bool = False) -> list[str]:
    """Падежные формы слова в свёртке, которых не находит его шаблон основы.

    noun=True — слово поля имён, которое приняли за короткое прилагательное («Ерофей», «Мокий»):
    у него ищутся и формы существительного.
    """
    if not (noun or needs_forms(word)):
        return []
    covered = re.compile(to_pattern(word), re.IGNORECASE)
    # Форма частого имени («Ксения» у «Ксений») — не слово дела: её не ищут.
    return [form for form in sorted(declined_forms(word)) if not covered.search(form) and form not in GIVEN_FORMS]


def form_patterns(word: str) -> list[str]:
    """Точные шаблоны падежных форм, которых не находит шаблон основы («Сызрани», «Квашей»)."""
    return [exact_pattern(form.capitalize()) for form in missing_forms(word)]


def plural_forms(name: str) -> set[str]:
    """Формы множественного числа имени в свёртке: «Олег» → «олеги», «олегов», «олегам»…

    Родительный падеж множественного числа («Александр» у «Александра», «Евгений» у «Евгения») и другие
    формы, совпавшие с формой другого частого имени, не берутся.
    """
    folded = fold(name)
    if folded in ("павел", "лев"):
        base, ends = folded[:-2] + ("ьв" if folded == "лев" else folded[-1]), ("ы", "ов", "ам", "ами", "ах")
    elif folded.endswith(("ий", "ей", "ай")):
        base, ends = folded[:-1], ("и", "ев", "ям", "ями", "ях")
    elif folded.endswith("ия"):
        base, ends = folded[:-1], ("и", "ям", "ями", "ях")  # «Евгений» — другое имя
    elif folded.endswith("ь"):
        base, ends = folded[:-1], ("и", "ей", "ям", "ями", "ях")
    elif folded.endswith("я"):
        base, ends = folded[:-1], ("и", "ям", "ями", "ях")
    elif folded.endswith(("а", "о")):
        base, ends = folded[:-1], ("ы", "ам", "ами", "ах")
    else:
        base, ends = folded, ("ы", "ов", "ам", "ами", "ах")
    forms = {re.sub("(?<=[гкхжшщч])ы", "и", base + end) for end in ends}
    return forms - (GIVEN_FORMS - name_forms(name))  # форма другого частого имени — не форма этого


def listed_name(entry: str) -> str | None:
    """Имя из списка частых имён, которое записано в extra_deny, или None для фамилии и редкого имени."""
    folded = fold(name_form(modern(entry.strip())))
    return next((name for name in GIVEN_NAMES if fold(name) == folded), None)


def entry_patronymics(entry: str) -> set[str]:
    """Отчества от записи extra_deny из списка частых имён: «Пётр» → «Петрович», «Петровна».

    Иначе список частых имён отсеял бы их, хотя само имя дело считает своим. Отчества от фамилии
    и редкого имени список и так не отсеивает.
    """
    name = listed_name(entry)
    return {(base if base.endswith("ич") else base + "а").capitalize()
            for base in patronymic_bases(name)} if name else set()


def entry_key(entry: str) -> str:
    entry = modern(entry.strip())
    return lexeme_key(name_form(entry)) if HAS_LETTER.search(entry) else entry.lower()


def read_rows(path: Path) -> tuple[str, list[dict[str, str]]]:
    """Возвращает имя первого столбца и строки реестра; нет файла — нет строк."""
    if not path.exists():
        return "", []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"повреждена строка CSV в {path}: полей больше, чем в шапке")
    return (reader.fieldnames or [""])[0], rows


def starts_sentence(before: str) -> bool:
    """Начинает ли слово новую фразу: начало поля, «!», «?» или точка после обычного слова."""
    before = before.rstrip()
    if not before or before[-1] in "!?":
        return True
    if before[-1] != ".":
        return False
    previous = LETTER_RUN.findall(before[-16:])
    return not previous or (len(previous[-1]) >= MIN_LETTERS and fold(previous[-1]) not in ABBREVIATIONS)


def caps_surname(value: str, end: int) -> bool:
    """Слово из заглавных — фамилия, если за ним идёт инициал или частое имя: «ТИХОМИРОВ Иван».

    Иначе это обычно сокращение вроде «ЗАГС» или «ЦАМО».
    """
    following = CAPS_NEXT.match(value, end)
    return following is not None and (following.group(1).endswith(".") or is_given_name(following.group(1)))


def name_tokens(value: str, every: bool) -> Iterator[str]:
    """Слова-имена поля и части слов через дефис.

    every=True — поле имён, мест или перечень: берётся каждое слово с заглавной,
    в том числе первое, после точки и из одних заглавных. every=False — свободный
    текст: слово в начале фразы и сокращение из заглавных вроде «ЗАГС» пропускаются.
    """
    for match in WORD.finditer(value):
        word = name_form(match.group(0))
        if sum(char.isalpha() for char in word) < MIN_LETTERS or is_name_abbreviation(word, value[match.end():]):
            continue
        if not every and (starts_sentence(value[: match.start()])
                          or match.group(0).isupper() and not caps_surname(value, match.end())):
            continue
        yield word
        yield from hyphen_parts(word)


def hyphen_parts(word: str) -> Iterator[str]:
    """Части слова через дефис.

    Часть с соединительным «о» («Хмурово-Каменка») возвращается с дефисом на конце: для неё
    строится шаблон только этой формы: её основа иначе могла бы совпасть с частым именем.
    """
    parts = word.split("-")
    if len(parts) > 1:
        for index, part in enumerate(parts):
            linking = index < len(parts) - 1 and part.endswith("о")
            if len(part) >= MIN_LETTERS and not (linking and is_given_name(part)):  # «Петро-»: частое имя
                yield part + "-" if linking else part


@dataclass(frozen=True)
class CaseWords:
    names: frozenset[str]        # поля имён: строчные слова реестров их не отсеивают
    places: frozenset[str]       # поля мест: так же
    free: frozenset[str]         # свободный текст
    free_common: frozenset[str]  # слова свободного текста, которые реестры пишут и со строчной
    ids: frozenset[str]
    numbers: frozenset[str] = frozenset()  # шифры и хеши из полей-ссылок
    weak: frozenset[str] = frozenset()     # мужские формы отчеств: бывают фамилиями («Вадимович»)
    fragments: frozenset[str] = frozenset()  # слова реестров, похожие на обрывок частого имени


def candidate_words(rows: list[dict[str, str]]) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Кандидаты из полей имён, мест и свободного текста; мужские формы отчеств; обрывки имён.

    Формы частых имён, женские отчества и служебные слова — не кандидаты. Мужская форма отчества
    бывает и фамилией: она остаётся кандидатом на просмотр. Обрывок имени («Христин») не получает
    шаблона — иначе ловил бы само имя, — но build называет его в предупреждении.
    """
    names: set[str] = set()
    places: set[str] = set()
    free: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if value:
                group = places if key in PLACE_FIELDS | LIST_FIELDS else names if key in NAME_FIELDS else free
                group.update(name_tokens(modern(value), every=group is not free))
    words = names | places | free
    weak = {word for word in words if patronymic_kind(word) == "male"}
    fragments = {word for word in words if given_name_kind(word) == "fragment"}
    common = {word for word in words if given_name_kind(word) or patronymic_kind(word) == "female"
              or fold(word) in FUNCTION_WORDS}
    return names - common, places - common, free - common, weak, fragments


def lowercase_words(rows: list[dict[str, str]]) -> set[str]:
    return {fold(word) for row in rows for key, value in row.items() if value and key not in QUERY_FIELDS
            for word in LOWER_WORD.findall(modern(value))}


def collect_case(case_dirs: Iterable[Path]) -> CaseWords:
    """Собирает слова-кандидаты и ID первых столбцов из всех реестров дел."""
    names: set[str] = set()
    places: set[str] = set()
    free: set[str] = set()
    lower: set[str] = set()
    ids: set[str] = set()
    numbers: set[str] = set()
    weak: set[str] = set()
    fragments: set[str] = set()
    for case_dir in case_dirs:
        for register in sorted(REGISTERS):
            key, rows = read_rows(Path(case_dir) / register)
            row_names, row_places, row_free, row_weak, row_fragments = candidate_words(rows)
            weak |= row_weak
            fragments |= row_fragments
            names |= row_names
            places |= row_places
            free |= row_free
            lower |= lowercase_words(rows)
            ids.update(value for row in rows if (value := modern(row.get(key) or "").strip()))
            numbers.update(number for row in rows for field in REFERENCE_FIELDS
                           for number in REFERENCE_NUMBER.findall(modern(row.get(field) or ""))
                           if is_reference_code(number))
    # Слово свободного текста, которое реестры пишут и со строчной («мельник»), скорее обычное слово,
    # но может быть и фамилией: оно идёт на просмотр (REVIEW), а не в блок.
    common = {word for word in free if fold(word) in lower}
    return CaseWords(frozenset(names), frozenset(places), frozenset(free), frozenset(common), frozenset(ids),
                     frozenset(numbers), frozenset(weak), frozenset(fragments))


class Reference:
    """Текст навыка, по которому шаблон проверяется без полного regex-поиска.

    Шаблон слова из одних букв находит текст, только если в тексте есть
    слово «основа + окончание из WORD_ENDINGS», поэтому хватает поиска во
    множестве слов. Тысячи regex-поисков по всему тексту шли минутами.
    """

    def __init__(self, text: str, id_text: str = "") -> None:
        """text — документы навыка; id_text — весь его текст с кодом и тестами, только для ID.

        Нумерация ID общая у всех дел и у тестов: ID, который уже есть
        в навыке, ничего не раскрывает. Слова кода и тестов на слова дела не влияют.
        """
        self.text = modern(text)
        self.id_text = modern(text + "\n" + id_text)
        words = LETTER_RUN.findall(self.text)
        self.words = {fold(word) for word in words}
        self.lowercase = {fold(word) for word in words if word.islower()}
        self.caps = {word for word in words if len(word) > 1 and word.isupper()}
        self.tokens = {token.lower() for token in ID_TOKEN.findall(self.id_text)}

    def has_word(self, word: str) -> bool:
        body = word_key(word)
        if not LETTER_RUN.fullmatch(body):
            return re.search(to_pattern(word), self.text, re.IGNORECASE) is not None
        return any(body + end in self.words for end in WORD_ENDINGS)

    def has_match(self, pattern: str) -> bool:
        return re.search(pattern, self.text, re.IGNORECASE) is not None

    def has_form(self, form: str) -> bool:
        """Точная форма слова (с конечным «ъ» или без) есть в тексте навыка — без regex-поиска.

        Форма через дефис («Ново-Сысоли») в множестве слов не лежит: её ищет regex.
        """
        folded = fold(form)
        if not LETTER_RUN.fullmatch(folded):
            return self.has_match(exact_pattern(form))
        return folded in self.words or folded + "ъ" in self.words

    def has_adjective(self, stem_: str) -> bool:
        body = fold(stem_)
        if not LETTER_RUN.fullmatch(body):
            return re.search(adjective_pattern(stem_), self.text, re.IGNORECASE) is not None
        return any(body + end + hard in self.words for end in ADJ_PATTERN_ENDINGS for hard in ("", "ъ"))

    def has_id(self, value: str) -> bool:
        if ID_TOKEN.fullmatch(value):
            return value.lower() in self.tokens
        return re.search(id_pattern(value), self.id_text, re.IGNORECASE) is not None


def allow_notes(names: Iterable[str], allow: Iterable[str], others: Iterable[str] = ()) -> list[str]:
    """Предупреждения: allow снимает слова полей имён, ни одна форма которых в allow не названа, и слова
    любых полей (others — поля мест и свободный текст), для которых запись — лишь падежная форма."""
    entries: dict[str, list[str]] = {}
    for entry in allow:
        entries.setdefault(entry_key(entry), []).append(entry.strip())
    forms = {fold(name_form(modern(entry.strip()))) for entry in allow}
    removed: dict[str, set[str]] = {}
    as_form: dict[str, set[str]] = {}  # «Валерия» — и женское имя, и родительный падеж «Валерий»
    names = set(names)
    for word in names:
        if (key := lexeme_key(word)) in entries:
            removed.setdefault(key, set()).add(word)
    # Обратный случай не отмечается: «Валерий» снимает «Валерия» — свой падеж, хотя это и женское имя.
    # Отличить их по написанию нельзя, а родительный падеж в реестрах встречается гораздо чаще.
    for word in names | set(others):
        if lexeme_key(word) in entries or fold(word) in forms:
            continue
        for entry in allow:
            _, entry_word, folded = allow_entry(entry)
            if not entry_word[:1].isupper():
                continue
            if folded in declined_forms(word) and fold(word) not in declined_forms(entry_word):
                as_form.setdefault(entry.strip(), set()).add(word)
    notes = [f"allow {', '.join(f'«{e}»' for e in sorted(entries[key]))} снимает и слова из полей имён: "
             f"{', '.join(f'«{w}»' for w in sorted(words))}. Если это фамилия, уберите запись из allow "
             "или впишите фамилию в extra_deny."
             for key, words in sorted(removed.items()) if not any(fold(word) in forms for word in words)]
    return notes + [f"allow «{entry}» — падежная форма и других слов реестров: "
                    f"{', '.join(f'«{w}»' for w in sorted(words))}, и снимает их. Если это другой человек "
                    "или фамилия, впишите слово в extra_deny; если то же слово — в allow."
                    for entry, words in sorted(as_form.items())]


def build_patterns(case_dirs: list[Path], *, extra_deny: Iterable[str] = (), allow: Iterable[str] = (),
                   reference_text: str | Reference = "") -> PatternSet:
    """Строит LEAK- и REVIEW-шаблоны по реестрам дела и тексту навыка.

    Запись allow с заглавной снимает все формы слова, со строчной — только эту форму; extra_deny сильнее allow.
    """
    case = collect_case(case_dirs)
    reference = reference_text if isinstance(reference_text, Reference) else Reference(reference_text)
    allow, extra_deny = list(allow), list(extra_deny)
    allowed = {entry_key(entry) for entry in allow}
    names, places, free, dropped = usable_words(case, reference)
    adjectives = {adjective for word in names if (adjective := adjective_stem(word))
                  and fold(adjective) not in allowed and not is_allowed(word, allowed, allow)}
    words = {word for word in names | places | free
             if not is_allowed(word, allowed, allow) and adjective_stem(word) not in adjectives}
    # Запись allow со строчной («ковали») снимает только эту форму, а не всё слово «Коваль».
    skip = {folded for _, word, folded in map(allow_entry, allow) if not word[:1].isupper()}
    # На просмотр, а не в блок: слово свободного текста, которое реестры пишут и со строчной,
    # и мужская форма отчества — скорее обычные слова, но бывают и фамилией.
    weak = (case.free_common - names - places) | case.weak
    found = word_findings(words, weak, reference, skip)
    # «Ерофей», «Мокий», «Сысой»: короткое имя на -ей, -ий, -ой похоже на прилагательное («синей»), поэтому
    # у слов полей имён с такой основой ловятся и формы существительного.
    found += [(exact_pattern(form.capitalize()), reference.has_form(form)) for word in names | places | free
              if fold(word).endswith(("ей", "ий", "ой")) and adjective_stem(word) in adjectives
              and not is_allowed(word, allowed, allow) for form in missing_forms(word, noun=True)
              if form not in skip]
    found += [(adjective_pattern(adjective), reference.has_adjective(adjective)) for adjective in adjectives]
    # «Живаго», «Бѣлаго»: у короткой основы «-аго» — и застывшая фамилия, и старый родительный
    # прилагательного. Слово ищется целиком (LEAK), а современные формы прилагательного — на просмотр.
    found += [(adjective_pattern(word[:-len(end)]), True) for word in names | places | free
              for end in OLD_ENDINGS if fold(word).endswith(end)
              and MIN_ADJ_STEM <= len(word) - len(end) < MIN_STEM and not is_allowed(word, allowed, allow)]
    found += [(id_pattern(value), reference.has_id(value)) for value in case.ids if value.lower() not in allowed]
    found += [(code_pattern(value), reference.has_id(value)) for value in case.numbers if value.lower() not in allowed]
    found += [(id_pattern(value), True) for value in next_ids(case.ids) - case.ids]
    common: dict[str, bool] = {}
    for pattern, seen in found:  # две формы дают один шаблон: блок, если хоть одна — не общая лексика
        common[pattern] = common.get(pattern, True) and seen
    deny = {pattern for pattern, seen in common.items() if not seen}
    patronymics = {word for entry in extra_deny for word in entry_patronymics(entry)}
    deny |= {pattern for entry in extra_deny for pattern in entry_patterns(entry)}
    deny |= {to_pattern(word) for word in patronymics}
    review = {pattern for pattern, seen in common.items() if seen} - deny
    leaked = {word for word in words if not word.endswith("-") and not common[to_pattern(word)]}
    leaked |= {name_form(modern(entry.strip())) for entry in extra_deny if HAS_LETTER.search(entry)} | patronymics
    notes = (allow_notes(case.names, allow, case.places | case.free)
             + dropped_notes(dropped | case.fragments, allowed, extra_deny))
    return PatternSet(tuple(sorted(deny)), tuple(sorted(review)), tuple(notes), latin_keys(leaked, reference))


def is_allowed(word: str, allowed: set[str], allow: list[str]) -> bool:
    """Слово снято allow: тот же ключ форм, падежная форма записи («Анатолия» снимает «Анатолий»; строчная
    запись — только если само слово — её форма) или запись попадает под шаблоны слова («Травец» — «Травца»)."""
    word = word.rstrip("-")
    if lexeme_key(word) in allowed:
        return True
    forms = declined_forms(word)
    entries = [allow_entry(entry) for entry in allow]
    for _, entry_word, folded in entries:
        # Строчная запись («ковали») — обычное слово: всё слово «Коваль» она не снимает.
        if entry_word[:1].isupper() and folded in forms or fold(word) in declined_forms(entry_word):
            return True
    patterns = [to_pattern(word)] + fleeting_patterns(word)
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns for text, _, _ in entries)


@lru_cache(maxsize=None)
def allow_entry(entry: str) -> tuple[str, str, str]:
    """Запись allow: текст в современном написании, слово в обычном написании и его свёртка."""
    text = modern(entry.strip())
    word = name_form(text)
    return text, word, fold(word)


def word_findings(words: Iterable[str], weak: set[str], reference: Reference,
                  skip: set[str] = frozenset()) -> list[tuple[str, bool]]:
    """Шаблоны слов с отметкой «есть в тексте навыка или слово слабое»; skip — снятые точные формы."""
    found: list[tuple[str, bool]] = []
    for word in words:
        if word.endswith("-"):  # соединительная часть топонима: только эта форма
            pattern = exact_pattern(word[:-1])
            found.append((pattern, reference.has_match(pattern)))
            continue
        seen = word in weak or reference.has_word(word)
        found.append((to_pattern(word), seen))
        found += [(pattern, seen or reference.has_match(pattern)) for pattern in fleeting_patterns(word)]
        found += [(exact_pattern(form.capitalize()), seen or reference.has_form(form))
                  for form in missing_forms(word) if form not in skip]
    return found


def usable_words(case: CaseWords, reference: Reference) -> tuple[set[str], set[str], set[str], set[str]]:
    """Слова имён, мест и свободного текста для шаблонов и отсеянные слова полей имён и мест.

    Слово, которое документы навыка пишут заглавными («ЦАМО», «ЗАГС»), — сокращение. Слово, которое
    они пишут со строчной («Книга», «Губернии», «Отец»), — обычное слово: шаблона нет. В полях имён и
    мест так может уйти и фамилия («Мельник», если документы пишут «мельник»), поэтому build называет
    все такие слова, пока их не впишут в allow или extra_deny.
    """
    caps = {word for word in case.names | case.places if word.rstrip("-").upper() in reference.caps}
    names, places = case.names - caps, case.places - caps
    free = {word for word in case.free if word.upper() not in reference.caps}
    dropped = {word for word in names | places if fold(word.rstrip("-")) in reference.lowercase}
    free = {word for word in free if fold(word) not in reference.lowercase}
    return names - dropped, places - dropped, free, dropped | caps


def dropped_notes(dropped: Iterable[str], allowed: set[str], extra_deny: Iterable[str]) -> list[str]:
    """Предупреждение об отсеянных словах реестров, о которых ещё нет решения в allow или extra_deny."""
    decided = allowed | {entry_key(entry) for entry in extra_deny}
    words = sorted({word.rstrip("-") for word in dropped if lexeme_key(word.rstrip("-")) not in decided})
    if not words:
        return []
    return [f"слова реестров не получили шаблонов: документы навыка пишут их со строчной или "
            f"заглавными, или это обрывок частого имени: "
            f"{', '.join(f'«{w}»' for w in words)}. Фамилию или название впишите в extra_deny, "
            "обычное слово — в allow."]


def merge_patterns(sets: Iterable[PatternSet]) -> PatternSet:
    """Объединяет шаблоны дел, собранные по отдельности: allow каждого дела действует только на него."""
    sets = list(sets)
    deny = {pattern for patterns in sets for pattern in patterns.deny}
    review = {pattern for patterns in sets for pattern in patterns.review} - deny
    latin = {key for patterns in sets for key in patterns.latin}
    return PatternSet(tuple(sorted(deny)), tuple(sorted(review)), tuple(note for p in sets for note in p.notes),
                      tuple(sorted(latin)))


def latin_key(text: str) -> str:
    """Латинский «скелет» слова: разные транслитерации дают один ключ («Tikhomirov» и «Tihomirov» — «tihomirov»)."""
    text = re.sub(r"ff$", "v", text.lower().replace("x", "ks"))  # «Plexinov», «Tikhomiroff»
    return LATIN_DIGRAPH.sub(lambda match: LATIN_DIGRAPHS[match.group(0)], text).translate(LATIN_LETTERS)


def ascii_letters(text: str) -> str:
    """Латиница без диакритики: «Černavskij» → «Cernavskij»."""
    return "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))


def cyrillic_latin_key(word: str) -> str:
    return latin_key("".join(TRANSLIT.get(char, char) for char in fold(word)))


LATIN_ENDINGS = frozenset({cyrillic_latin_key(end) for end in STEM_ENDINGS if end not in OLD_ENDINGS} | {"", "s", "es"})


def latin_keys(words: Iterable[str], reference: Reference | None = None) -> tuple[str, ...]:
    """Ключи латинского написания для слов LEAK: ключ не короче MIN_LATIN_KEY букв.

    Слово через дефис ключа не даёт: его части — отдельные слова. Ключ, который уже находит
    латиницу текста навыка («Telegram»), не берётся: это общее слово, а не данные дела.
    """
    keys = {key for word in words if "-" not in word and len(key := cyrillic_latin_key(stem(word))) >= MIN_LATIN_KEY}
    if reference is not None:
        seen = LatinMatcher(keys)
        keys -= {hit.pattern.removeprefix("latin:") for hit in seen.hits(0, reference.text)}
    return tuple(sorted(keys))


class LatinMatcher:
    """Латинское написание слов дела в английском тексте: ключ основы и окончание из LATIN_ENDINGS."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.by_head: dict[str, list[str]] = {}
        for key in keys:
            self.by_head.setdefault(key[:3], []).append(key)

    def hits(self, line_no: int, text: str) -> list[Hit]:
        found = []
        for token in set(LATIN_TOKEN.findall(text)):
            word = latin_key(token if token.isascii() else ascii_letters(token))
            for key in self.by_head.get(word[:3], ()):
                if word.startswith(key) and word[len(key):] in LATIN_ENDINGS:
                    found.append(Hit("LEAK", line_no, "latin:" + key, text, token))
                    break
        return found


def git_path(text: str) -> str:
    """Путь из заголовка diff. Путь в кавычках git пишет с восьмеричными кодами байтов UTF-8."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    return text


def new_path(line: str) -> str:
    """Путь после изменения из «diff --git a/X b/Y» или «Binary files X and Y differ»; /dev/null — пусто."""
    if line.startswith("diff --git "):
        rest = line[len("diff --git "):]
        if rest.endswith('"'):
            start = rest.rfind(' "') + 1
        elif " b/" in rest:
            start = rest.rfind(" b/") + 1
        elif len(rest) % 2 == 1 and rest[len(rest) // 2] == " ":  # равные половины: --no-prefix, c/ и w/
            start = len(rest) // 2 + 1
        else:
            start = rest.rfind(" ") + 1
    else:
        rest = line[len("Binary files "):].removesuffix(" differ")
        start = rest.rfind(" and ") + len(" and ") if " and " in rest else len(rest)
    path = git_path(rest[start:]) if start > 0 else ""
    return "" if path == "/dev/null" else path


def diff_lines(lines: Iterable[tuple[int, str]]) -> Iterator[tuple[int, str]]:
    """Добавленные строки diff и пути файлов после изменения.

    Внутри блока @@ строки считаются по его размерам, поэтому добавленная
    строка «++ …» не путается с заголовком «+++ путь». Строки «+» вне
    блоков тоже проверяются: так читается diff без заголовков @@. Старые
    пути (удаление, переименование) не проверяются: иначе нельзя убрать
    утёкшее имя файла. Вход без единого признака diff или с цветом — ошибка,
    а не «проверено 0 строк».
    """
    old_left = new_left = 0
    path, announced, structured, seen = "", False, False, False
    for line_no, line in lines:
        seen = seen or bool(line.strip())
        if "\x1b[" in line:
            raise ValueError("в diff есть цветовые коды: передайте git diff --no-color")
        if old_left > 0 or new_left > 0:
            mark = line[:1]
            if mark == "+":
                new_left -= 1
                yield line_no, line[1:]
            elif mark == "-":
                old_left -= 1
            elif mark != "\\":
                old_left, new_left = old_left - 1, new_left - 1
            continue
        structured = structured or line.startswith(("diff --git ", "@@ ", "+", "--- ", "Binary files "))
        if hunk := HUNK.match(line):
            old_left, new_left = int(hunk.group(1) or 1), int(hunk.group(2) or 1)
        elif line.startswith("diff --git "):
            path, announced = new_path(line), False
        elif line.startswith("new file mode") and path:
            announced = True
            yield line_no, path  # у пустого нового файла путь есть только в заголовке
        elif line.startswith("+++ "):
            if not announced and (target := git_path(line[4:])) != "/dev/null":
                yield line_no, target
        elif line.startswith(("rename to ", "copy to ")):
            yield line_no, git_path(line.split(" ", 2)[2])
        elif line.startswith("Binary files "):
            if not announced and (target := new_path(line)):
                yield line_no, target
        elif line.startswith("+"):
            yield line_no, line[1:]
    if seen and not structured:
        raise ValueError("во входе --diff нет ни заголовков diff, ни добавленных строк «+»")


def checked_lines(lines: Iterable[str], diff: bool) -> Iterator[tuple[int, str]]:
    """Строки для проверки в современной орфографии; в режиме diff — только новое."""
    numbered = ((line_no, modern(line.rstrip("\r"))) for line_no, line in enumerate(lines, 1))
    return diff_lines(numbered) if diff else numbered


FOLD_EXTRA = str.maketrans({"ё": "е", "ı": "i", "ſ": "s"})  # «ı» и «ſ» regex без учёта регистра считает за «i» и «s»


def scan_fold(text: str) -> str:
    """Свёртка для отбора шаблонов: совпадение regex с IGNORECASE сохраняется в ней как подстрока."""
    return "".join(char.lower()[:1] for char in text).translate(FOLD_EXTRA)


def literal_head(source: str, size: int = 3) -> str:
    """Первые size букв тела шаблона в виде scan_fold; пусто, если тело начинается не с букв."""
    for prefix in (f"(^|[^{LETTER}])", r"(?<![\w-])"):
        if source.startswith(prefix):
            body = source[len(prefix):]
            break
    else:
        return ""
    chars: list[str] = []
    index = 0
    while len(chars) < size and index < len(body):
        if body.startswith("[её]", index):
            chars.append("е")
            index += 4
        elif body[index] == "\\" and index + 1 < len(body) and not body[index + 1].isalnum():
            chars.append(body[index + 1])
            index += 2
        elif body[index] in ".^$*+?{}[]()|\\":
            return ""
        else:
            chars.append(body[index])
            index += 1
    return scan_fold("".join(chars)) if len(chars) == size else ""


class Matcher:
    """Шаблоны, разложенные по первым трём буквам тела.

    Строку проверяют только шаблоны, чьи три первые буквы в ней есть: без них
    шаблон совпасть не может, поэтому результат тот же, что у полного перебора,
    а regex-поисков во много раз меньше. Шаблоны другой формы проверяются всегда.
    """

    def __init__(self, patterns: PatternSet) -> None:
        self.by_head: dict[str, list[int]] = {}
        self.always: list[int] = []
        sources = [("LEAK", source) for source in patterns.deny] + [("REVIEW", source) for source in patterns.review]
        self.entries = [(kind, source, re.compile(source, re.IGNORECASE)) for kind, source in sources]
        for index, (_, source, _) in enumerate(self.entries):
            head = literal_head(source)
            (self.by_head.setdefault(head, []) if head else self.always).append(index)

    def hits(self, line_no: int, text: str) -> list[Hit]:
        folded = scan_fold(text)
        heads = {folded[start:start + 3] for start in range(len(folded) - 2)} & self.by_head.keys()
        chosen = sorted(self.always + [index for head in heads for index in self.by_head[head]])
        found = []
        for kind, source, regex in (self.entries[index] for index in chosen):
            match = regex.search(text)
            if match:
                found.append(Hit(kind, line_no, source, text, match.group(0).strip()))
        return found


def scan_numbered(numbered: Iterable[tuple[int, str]], patterns: PatternSet) -> list[Hit]:
    """Совпадения по строкам; один фрагмент строки печатается один раз, LEAK раньше REVIEW."""
    matcher, latin = Matcher(patterns), LatinMatcher(patterns.latin)
    hits: list[Hit] = []
    for line_no, text in numbered:
        fragments: set[str] = set()
        found = matcher.hits(line_no, text) + latin.hits(line_no, text)
        for hit in sorted(found, key=lambda hit: hit.kind != "LEAK"):
            if hit.fragment not in fragments:
                fragments.add(hit.fragment)
                hits.append(hit)
    return hits


def scan_lines(lines: Iterable[str], patterns: PatternSet, *, diff: bool = False) -> list[Hit]:
    return scan_numbered(checked_lines(lines, diff), patterns)


def load_config(case_dir: Path) -> tuple[list[str], list[str]]:
    path = case_dir / CONFIG_NAME
    if not path.exists():
        return [], []
    config = read_json(path)
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ValueError(f"{path}: ожидается объект только с ключами {', '.join(sorted(CONFIG_KEYS))}")
    lists = [config.get(key, []) for key in ("extra_deny", "allow")]
    if not all(isinstance(items, list) and all(isinstance(item, str) and item.strip() for item in items)
               for items in lists):
        raise ValueError(f"{path}: extra_deny и allow — списки непустых строк")
    return lists[0], lists[1]


def reference_text(root: Path | None, *, everything: bool = False) -> str:
    """Документы навыка (.md, .yaml вне tests/); everything=True — все его текстовые файлы, для ID."""
    if root is None:
        return ""
    suffixes, skipped = (ID_SOURCE_SUFFIXES, {".git"}) if everything else (REFERENCE_SUFFIXES, REFERENCE_SKIPPED_DIRS)
    files = sorted(path for path in reference_files(root) if path.suffix in suffixes
                   and not skipped & set(path.relative_to(root).parts))
    return "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in files)


def reference_files(root: Path) -> list[Path]:
    """Файлы навыка: в рабочей копии git — только отслеживаемые; иначе все, кроме папок с реестрами дел."""
    try:
        listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
        names = [name for name in listed.decode("utf-8").split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        names = []
    if "SKILL.md" in names:
        return [root / name for name in names if (root / name).is_file()]
    # Без git (или копия не отслеживается): файлы в корне и в папках навыка, кроме папок с реестрами дел.
    files = [path for path in root.iterdir() if path.is_file()]
    files += [path for folder in SKILL_FOLDERS if (root / folder).is_dir()
              for path in (root / folder).rglob("*") if path.is_file()]
    case_folders = {path.parent for path in files if path.name in REGISTERS}
    return [path for path in files if not case_folders & set(path.parents)]


def build_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, list[Path]]:
    """Проверяет пути build до чтения реестров: файл шаблонов не должен попасть в репозиторий."""
    out = outside_skill(args.out)
    reference = Path(args.reference_dir).expanduser().resolve() if args.reference_dir else None
    if reference is not None and not (reference / "SKILL.md").is_file():
        raise ValueError(f"{reference}: --reference-dir — папка навыка с SKILL.md")
    if reference is not None and (out == reference or reference in out.parents):
        raise ValueError("файл шаблонов содержит данные дела: пишите его вне --reference-dir")
    case_dirs = [Path(value).expanduser() for value in args.case_dir]
    for case_dir in case_dirs:
        if not any((case_dir / register).is_file() for register in REGISTERS):
            raise ValueError(f"{case_dir}: нет папки дела или реестров в ней")
        if reference is not None and reference in case_dir.resolve().parents:
            raise ValueError(f"{case_dir}: папка дела внутри --reference-dir — её слова стали бы «текстом навыка»")
    return out, reference, case_dirs


def run_build(args: argparse.Namespace) -> int:
    out, reference, case_dirs = build_inputs(args)
    text = Reference(reference_text(reference), id_text=reference_text(reference, everything=True))
    sets = []
    for case_dir in case_dirs:  # allow каждого дела действует только на его слова
        extra_deny, allow = load_config(case_dir)
        sets.append(build_patterns([case_dir], extra_deny=extra_deny, allow=allow, reference_text=text))
    patterns = merge_patterns(sets)
    for note in patterns.notes:
        print(f"Предупреждение: {note}", file=sys.stderr)
    if not patterns.deny:
        raise ValueError("ни одного шаблона LEAK: проверьте --case-dir и реестры")
    save_json(out, {"deny": list(patterns.deny), "review": list(patterns.review), "latin": list(patterns.latin)})
    print(f"deny: {len(patterns.deny)} patterns; review: {len(patterns.review)} patterns")
    return 0


def load_patterns(path: str) -> PatternSet:
    data = read_json(path)
    lists = [data.get(key) for key in ("deny", "review")] + [data.get("latin", [])] if isinstance(data, dict) else [None]
    if not all(isinstance(items, list) and all(isinstance(item, str) for item in items) for items in lists):
        raise ValueError(f"{path}: ожидается объект со списками строк deny, review и latin")
    # синтаксис regex проверяет Matcher при компиляции
    return PatternSet(tuple(lists[0]), tuple(lists[1]), latin=tuple(lists[2]))


def run_check(args: argparse.Namespace) -> int:
    patterns = load_patterns(args.patterns)
    if args.input == "-":
        text = sys.stdin.buffer.read().decode("utf-8-sig")
    else:
        text = Path(args.input).expanduser().read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("вход пуст: проверять нечего — проверьте команду, которая готовит текст или diff")
    # Только «\n»: splitlines() режет и по U+2028 и \x0c, и в diff хвост строки терялся бы.
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    numbered = list(checked_lines(lines, args.diff))
    hits = scan_numbered(numbered, patterns)
    for hit in hits:
        print(f"{hit.kind:<6} line {hit.line_no}: «{hit.fragment}» in: {hit.text.strip()[:120]}")
    leaks = sum(hit.kind == "LEAK" for hit in hits)
    print(f"checked lines: {len(numbered)}; leaks: {leaks}; review: {len(hits) - leaks}")
    return 1 if leaks else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка исходящего текста по реестрам дела.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Собрать шаблоны из реестров дела")
    build.add_argument("--case-dir", action="append", required=True, help="Директория дела; можно повторять")
    build.add_argument("--reference-dir",
                       help="Папка навыка без проверяемых правок: её слова идут в REVIEW, а не в LEAK")
    build.add_argument("--out", required=True, help="JSON с шаблонами (вне папки навыка и --reference-dir)")
    check = commands.add_parser("check", help="Проверить текст или diff по шаблонам")
    check.add_argument("--patterns", required=True, help="JSON, собранный командой build")
    check.add_argument("--diff", action="store_true",
                       help="Проверять только добавленные строки diff и пути файлов после изменения")
    check.add_argument("input", nargs="?", default="-", help="Файл с текстом; «-» или ничего — stdin")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_build(args) if args.command == "build" else run_check(args)
    except (OSError, ValueError, re.error, csv.Error, SourceError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
