"""Тесты privacy_scan.py на синтетических реестрах с вымышленными именами.

Запуск: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.privacy_scan import (MIN_STEM, Matcher, PatternSet, Reference, adjective_pattern, build_patterns,
                                    merge_patterns, modern, reference_text, scan_lines, to_pattern)
from tests import bench_privacy_scan

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
SKILL_ROOT = SCRIPTS.parent


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(case_dir: Path, names: str = "", summary: str = "") -> None:
    write_csv(case_dir / "artifact-register.csv", ["artifact_id", "names_as_written", "summary"],
              [{"artifact_id": "A-9001", "names_as_written": names, "summary": summary}])


def kinds(patterns: PatternSet, text: str, diff: bool = False) -> list[str]:
    return [hit.kind for hit in scan_lines(text.split("\n"), patterns, diff=diff)]


class BuildPatternsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.case = Path(tmp.name)

    def patterns(self, names: str = "", summary: str = "", **options: object) -> PatternSet:
        write_artifacts(self.case, names, summary)
        return build_patterns([self.case], **options)

    def test_ye_and_yo_match_each_other(self) -> None:
        patterns = self.patterns(names="Клёнов")
        self.assertEqual(kinds(patterns, "Кленов"), ["LEAK"])
        self.assertEqual(kinds(patterns, "КЛЁНОВА"), ["LEAK"])
        self.assertTrue(any("[её]" in pattern for pattern in patterns.deny))
        self.assertFalse(any("[е[её]]" in pattern for pattern in patterns.deny))
        self.assertEqual(kinds(self.patterns(names="Кленов"), "Клёнов"), ["LEAK"])

    def test_capital_yo_matches_ye(self) -> None:
        self.assertEqual(kinds(self.patterns(names="Ёршиков"), "Ершиков"), ["LEAK"])
        self.assertEqual(kinds(self.patterns(extra_deny=["ЁРШИКОВА"]), "Ершиков"), ["LEAK"])
        patterns = self.patterns(names="Ёршиков", reference_text="Пример: Ершиков.")
        self.assertEqual(kinds(patterns, "Ёршиков"), ["REVIEW"])

    def test_decomposed_letters_are_normalized(self) -> None:
        patterns = self.patterns(names="Клёнов")
        self.assertEqual(kinds(patterns, "Клёнов"), ["LEAK"])

    def test_pattern_is_anchored_at_word_start(self) -> None:
        patterns = self.patterns(names="Ранов")
        self.assertEqual(kinds(patterns, "Баранова"), [])
        self.assertEqual(kinds(patterns, "у Рановых"), ["LEAK"])

    def test_pattern_needs_a_case_ending_at_word_end(self) -> None:
        patterns = self.patterns(names="Тихомирова Каменка")
        self.assertEqual(kinds(patterns, "Тихомировъ\nТихомировым\nТихомировымъ\nКаменкой"), ["LEAK"] * 4)
        self.assertEqual(kinds(patterns, "Тихомировский Каменкин"), [])

    def test_reference_word_sharing_only_a_prefix_stays_leak(self) -> None:
        patterns = self.patterns(names="Тихомирова", reference_text="Тихомировский район.")
        self.assertEqual(kinds(patterns, "Тихомирова"), ["LEAK"])

    def test_reference_word_with_same_ending_does_not_move_to_review(self) -> None:
        patterns = self.patterns(names="Ранов", reference_text="Пример: Баранов.")
        self.assertEqual(kinds(patterns, "Ранов"), ["LEAK"])

    def test_case_forms_are_reduced_to_stem(self) -> None:
        patterns = self.patterns(summary="Жили в семье Тихомировых, в Каменке, писали Залесскому.")
        for word in ("Тихомиров", "Каменка", "Залесский"):
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])

    def test_longest_ending_decides_the_stem(self) -> None:
        patterns = self.patterns(names="Хмурая Сонное")
        self.assertEqual(kinds(patterns, "Хмурая\nСонное"), ["LEAK", "LEAK"])
        self.assertEqual(kinds(patterns, "Хмуракова, хмура, сонно"), [])

    def test_old_spelling_matches_modern_spelling(self) -> None:
        patterns = self.patterns(names="Бѣлоцвѣтовъ Ѳеодоръ")
        self.assertEqual(kinds(patterns, "Белоцветов"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Бѣлоцвѣтовъ"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Федор, Феодор, Ѳеодоръ"), [])

    def test_short_word_with_final_hard_sign_is_not_candidate(self) -> None:
        self.assertEqual(kinds(self.patterns(names="Сынъ Бѣлоцвѣтовъ"), "сын, сыновья"), [])

    def test_given_names_and_patronymics_are_not_candidates(self) -> None:
        patterns = self.patterns(names="Ольга Ивановна Олегович Юрьевич Никитична",
                                 summary="Записана с Ольгой Петровной и Петром")
        self.assertEqual(kinds(patterns, "Ольгой Петровной, Ольга Ивановна, Петром, Никитична"), [])
        self.assertEqual(kinds(patterns, "Олегович\nЮрьевич"), ["REVIEW"] * 2)

    def test_vich_surnames_are_candidates(self) -> None:
        patterns = self.patterns(names="Ракитевич Ольхович Зябич")
        for word in ("Ракитевич", "Ольхович", "Зябич"):
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])

    def test_patronymic_must_be_built_from_a_whole_given_name(self) -> None:
        common = "Павловна Михайловна Васильевна Никитична Николаевна"
        male = "Иаковлевич Эдуардович Игоревич Ильич"
        patterns = self.patterns(names=f"{common} {male} Павле")
        self.assertEqual(kinds(patterns, "\n".join(common.split() + ["Павле"])), [])
        self.assertEqual(kinds(patterns, "\n".join(male.split())), ["REVIEW"] * 4)
        surnames = "Андрич Борович Марович Петрич Иванич Никич Симич Анич"
        patterns = self.patterns(names=surnames)
        for word in surnames.split():
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])

    def test_genitive_of_names_ending_in_short_i_is_given_name(self) -> None:
        patterns = self.patterns(names="Василия Георгия Григория Тихомирова")
        self.assertEqual(kinds(patterns, "Василия\nГеоргия\nГригория"), [])
        self.assertEqual(kinds(patterns, "Тихомиров"), ["LEAK"])

    def test_name_like_word_does_not_get_adjective_stem(self) -> None:
        patterns = self.patterns(names="Фомий Белая")
        self.assertEqual(kinds(patterns, "Фомой"), [])
        self.assertEqual(kinds(patterns, "Фомий\nБелой"), ["LEAK"] * 2)

    def test_given_names_in_eyu_form_are_not_candidates(self) -> None:
        patterns = self.patterns(names="с Натальею и Ксениею Белоцветовой")
        self.assertEqual(kinds(patterns, "Наталья\nКсения\nБелоцветова"), ["LEAK"])

    def test_abbreviated_given_names_are_not_candidates(self) -> None:
        patterns = self.patterns(names="Георг. Конст. Праск.")
        self.assertEqual(kinds(patterns, "Георгий, Прасковья; Георг. Конст. Праск."), [])

    def test_name_prefix_is_a_candidate_unless_an_ending_makes_it_a_given_name(self) -> None:
        patterns = self.patterns(names="Вален Констан Евген")
        self.assertEqual(kinds(patterns, "Вален\nКонстан"), ["LEAK", "LEAK"])
        self.assertEqual(kinds(patterns, "Евген\nЕвгений"), [])

    def test_caps_surname_in_name_field_is_candidate(self) -> None:
        patterns = self.patterns(names="ТИХОМИРОВ ИВАН ПЕТРОВИЧ; КЛЁНОВ, Иван")
        self.assertEqual(kinds(patterns, "Тихомиров\nКленов"), ["LEAK", "LEAK"])
        self.assertEqual(kinds(patterns, "Иван"), [])
        self.assertEqual(kinds(patterns, "Петрович"), ["REVIEW"])

    def test_stress_marks_and_soft_hyphens_are_ignored(self) -> None:
        patterns = self.patterns(names="Тихоми\u00adро\u0301в")
        self.assertEqual(kinds(patterns, "Тихомиров\nТихоми\u00adро\u0301ва"), ["LEAK", "LEAK"])

    def test_hyphenated_name_parts_get_own_patterns(self) -> None:
        patterns = self.patterns(names="Ранов-Гончаров")
        self.assertEqual(kinds(patterns, "Гончаров\nу Рановой"), ["LEAK", "LEAK"])
        self.assertIn("LEAK", kinds(patterns, "Ранов-Гончаров"))

    def test_joining_vowel_part_is_not_a_word(self) -> None:
        patterns = self.patterns(names="Сосново-Каменка")
        self.assertEqual(kinds(patterns, "Каменка"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Сосновый"), [])

    def test_short_adjective_surname_matches_case_forms(self) -> None:
        patterns = self.patterns(names="Белая Седой")
        self.assertEqual(kinds(patterns, "Белой\nБелую\nСедого\nСедым"), ["LEAK"] * 4)
        self.assertEqual(kinds(patterns, "Белкин, Белоус, Седельников"), [])

    def test_short_adjective_place_keeps_narrow_pattern(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "places"],
                  [{"person_id": "P-9001", "places": "Сонное"}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "Сонное"), ["LEAK"])
        self.assertEqual(kinds(patterns, "по сонному"), [])

    def test_name_field_word_lowercase_elsewhere_is_kept(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "preferred_label", "occupation_or_status"],
                  [{"person_id": "P-9001", "preferred_label": "Мельник Хмурова", "occupation_or_status": "мельник"}])
        self.assertEqual(kinds(build_patterns([self.case]), "Мельник"), ["LEAK"])
        patterns = build_patterns([self.case], reference_text="Отец был мельником.")
        self.assertEqual(kinds(patterns, "Мельник"), ["REVIEW"])
        write_artifacts(self.case, summary="Свидетель — сосед Мельник.")
        self.assertEqual(kinds(build_patterns([self.case]), "Мельник"), ["LEAK"])

    def test_one_fragment_is_reported_once_and_leak_wins(self) -> None:
        patterns = PatternSet((to_pattern("Тихомиров"),), (r"(?<![\w-])Тихомиров", to_pattern("Тихомиров") + "?"))
        self.assertEqual(kinds(patterns, "Тихомиров"), ["LEAK"])

    def test_place_fields_skip_given_name_forms_unless_extra_deny(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "places"],
                  [{"person_id": "P-9001", "places": "Николы, Ветлуга; Виктор; Никола"}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "Ветлуга"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Николы\nВиктор\nНикола"), [])
        self.assertEqual(kinds(build_patterns([self.case], extra_deny=["Николы"]), "Николы"), ["LEAK"])

    def test_caps_word_written_in_caps_by_reference_text_is_not_candidate(self) -> None:
        patterns = self.patterns(names="ТИХОМИРОВ (ЦАМО)", reference_text="запрос в ЦАМО")
        self.assertEqual(kinds(patterns, "ЦАМО"), [])
        self.assertEqual(kinds(patterns, "Тихомиров"), ["LEAK"])

    def test_relatives_jurisdictions_and_query_keep_first_word(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "relatives_as_reported", "historical_jurisdictions"],
                  [{"person_id": "P-9001", "relatives_as_reported": "Гончарова Мария (мать)",
                    "historical_jurisdictions": "Ветлуга, уезд"}])
        write_csv(self.case / "search-log.csv", ["search_id", "query"],
                  [{"search_id": "S-9001", "query": "Хмурова 1890"}])
        patterns = build_patterns([self.case])
        for word in ("Гончарова", "Ветлуга", "Хмурова"):
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])
        self.assertEqual(kinds(patterns, "Мария"), [])

    def test_list_field_first_word_lowercase_elsewhere_is_common(self) -> None:
        write_csv(self.case / "search-log.csv", ["search_id", "query", "next_step"],
                  [{"search_id": "S-9001", "query": "Перед Хмуровой", "next_step": "перед отправкой"}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "Перед"), [])
        self.assertEqual(kinds(patterns, "Хмуровой"), ["LEAK"])

    def test_name_field_word_lowercase_in_reference_text_is_common(self) -> None:
        patterns = self.patterns(names="Того же села Хмурова", reference_text="после того")
        self.assertEqual(kinds(patterns, "того"), [])
        self.assertEqual(kinds(patterns, "Хмурова"), ["LEAK"])

    def test_surname_like_word_lowercase_in_reference_text_is_reported(self) -> None:
        patterns = self.patterns(names="Казаков Книга Хмурова", reference_text="список казаков; книга")
        self.assertEqual(kinds(patterns, "Казаков\nКнига"), [])
        self.assertEqual(kinds(patterns, "Хмурова"), ["LEAK"])
        self.assertEqual(len(patterns.notes), 1, patterns.notes)
        self.assertIn("«Казаков»", patterns.notes[0])
        self.assertIn("«Книга»", patterns.notes[0])
        self.assertEqual(self.patterns(names="Казаков", reference_text="казаков", allow=["Казаков"]).notes, ())
        patterns = self.patterns(names="Казаков", reference_text="казаков", extra_deny=["Казаков"])
        self.assertEqual(kinds(patterns, "Казаковым"), ["LEAK"])

    def test_search_query_and_filters_do_not_hide_free_text_names(self) -> None:
        write_artifacts(self.case, summary="Свидетель — сосед Хмуров из Залесья.")
        write_csv(self.case / "search-log.csv", ["search_id", "query", "filters"],
                  [{"search_id": "S-9001", "query": "хмуров залесье 1890", "filters": "место=залесья"}])
        self.assertEqual(kinds(build_patterns([self.case]), "Хмуров\nЗалесья"), ["LEAK"] * 2)

    def test_words_after_register_abbreviations_are_candidates(self) -> None:
        patterns = self.patterns(summary="Жила в респ. Хмуровия, затем колх. Залесье, прож. Кривоборье; "
                                         "крещ. Ольшанки, посел. Зябликово")
        for word in ("Хмуровия", "Залесье", "Кривоборье", "Ольшанки", "Зябликово"):
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])

    def test_latin_extra_deny_matches_word_beginnings(self) -> None:
        patterns = self.patterns(extra_deny=["Tikhomirov"])
        self.assertEqual(kinds(patterns, "Tikhomirova\nTikhomirovs\ntikhomirov's"), ["LEAK"] * 3)
        self.assertEqual(kinds(patterns, "XTikhomirov"), [])

    def test_allow_covers_every_form_and_reports_other_name_words(self) -> None:
        patterns = self.patterns(names="Ивина Ивиной", allow=["Ивина"])
        self.assertEqual(kinds(patterns, "Ивина\nИвиной\nИвиным"), [])
        self.assertEqual(patterns.notes, ())
        patterns = self.patterns(names="Белая", allow=["Белое"])
        self.assertEqual(kinds(patterns, "Белой"), [])
        self.assertEqual(len(patterns.notes), 1, patterns.notes)
        self.assertIn("Белая", patterns.notes[0])
        self.assertIn("Белое", patterns.notes[0])
        patterns = self.patterns(names="Белая", allow=["Белое"], extra_deny=["Белая"])
        self.assertEqual(kinds(patterns, "Белой\nБелого"), ["LEAK"] * 2)

    def test_allow_of_a_name_covers_its_case_forms(self) -> None:
        patterns = self.patterns(names="Анатолий Анатолия Анатолию Анатолием Хмурова", allow=["Анатолий"])
        self.assertEqual(kinds(patterns, "Анатолий\nАнатолия\nАнатолию\nАнатолием"), [])
        self.assertEqual(kinds(patterns, "Хмурова"), ["LEAK"])

    def test_case_forms_of_soft_short_and_ii_words_are_caught(self) -> None:
        patterns = self.patterns(names="Анатолий Коваль Сызрань Поречие Кваша Агния Бугай Ерофей Пантелей")
        text = ("Анатолия\nАнатолию\nАнатолием\nКоваля\nКовалем\nСызрани\nСызранью\nПоречия\nПоречием\n"
                "Кваши\nКвашей\nАгнии\nАгнией\nБугая\nБугаем\nЕрофея\nЕрофеем\nПантелея\nПантелеем")
        self.assertEqual(kinds(patterns, text), ["LEAK"] * 19)
        self.assertEqual(kinds(patterns, "Ковалёв\nСызранский\nАнатольевка"), [])
        self.assertFalse(any("Квашы" in pattern for pattern in patterns.deny), "после ш пишется «и»")

    def test_case_form_already_in_skill_text_goes_to_review(self) -> None:
        patterns = self.patterns(names="Кваша", reference_text="у кваши")
        self.assertEqual(kinds(patterns, "Кваши"), ["REVIEW"])
        self.assertEqual(kinds(patterns, "Квашей"), ["LEAK"])

    def test_extra_deny_catches_case_forms_of_soft_and_short_words(self) -> None:
        patterns = self.patterns(extra_deny=["Анатолий", "Кваша", "Сызрань"])
        self.assertEqual(kinds(patterns, "Анатолия\nКвашей\nСызрани"), ["LEAK"] * 3)

    def test_fleeting_vowel_after_yo_is_found(self) -> None:
        patterns = self.patterns(names="Орёл Хмурова")
        self.assertEqual(kinds(patterns, "Орла\nОрлом\nОрёл"), ["LEAK"] * 3)

    def test_apostrophe_variants_match_each_other(self) -> None:
        for written, other in (("Сем'янчук", "Сем’янчук\nСемʼянчук"), ("Сем’янчук", "Сем'янчук")):
            with self.subTest(written):
                self.assertEqual(kinds(self.patterns(names=written), other), ["LEAK"] * len(other.split("\n")))
        self.assertEqual(kinds(self.patterns(extra_deny=["Сем'янчук"]), "Семʼянчук"), ["LEAK"])

    def test_free_text_words_that_look_like_name_fragments_are_reported(self) -> None:
        patterns = self.patterns(summary="Жили у Мариных, потом в селе Марино.")
        self.assertEqual(len(patterns.notes), 1, patterns.notes)
        self.assertIn("«Мариных»", patterns.notes[0])
        self.assertIn("«Марино»", patterns.notes[0])

    def test_old_spelling_adjective_endings_match_modern_ones(self) -> None:
        patterns = self.patterns(names="Звенигородский Белевскія")
        self.assertEqual(kinds(patterns, "Звенигородскаго\nБелевская\nБелевской\nБелевские"), ["LEAK"] * 4)
        modern_register = self.patterns(names="Белевская Звенигородский")
        self.assertEqual(kinds(modern_register, "Белевскія\nЗвенигородскаго"), ["LEAK"] * 2)

    def test_frozen_genitive_surnames_keep_their_old_ending(self) -> None:
        patterns = self.patterns(names="Живаго Бураго Хмурова")
        self.assertEqual(kinds(patterns, "Живаго\nБураго"), ["LEAK"] * 2)
        # «-аго» у короткой основы — и застывшая фамилия, и старый родительный прилагательного:
        # современные формы прилагательного идут на просмотр, а не в блок.
        self.assertEqual(kinds(patterns, "живой\nЖивого\nбурый\nв буром"), ["REVIEW"] * 4)
        self.assertEqual(patterns.notes, ())

    def test_short_old_spelling_adjective_surname_flags_modern_forms(self) -> None:
        patterns = self.patterns(names="Бѣлаго Хмурова")
        self.assertEqual(kinds(patterns, "Белаго"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Белый\nБелого"), ["REVIEW"] * 2)

    def test_allow_of_a_modern_adjective_does_not_remove_a_frozen_surname(self) -> None:
        patterns = self.patterns(names="Живаго Хмурова", allow=["Живой"])
        self.assertEqual(kinds(patterns, "Живаго"), ["LEAK"])

    def test_case_word_whose_stem_plus_ia_is_in_skill_text_stays_leak(self) -> None:
        patterns = self.patterns(names="Бессараб Мордова Карел Хмурова",
                                 reference_text="Бессарабия, Мордовия, Карелия; Bessarabia, Mordovia, Karelia.")
        self.assertEqual(kinds(patterns, "Бессараб\nМордова\nКарел"), ["LEAK"] * 3)
        self.assertEqual(kinds(patterns, "Letters of Ivan Bessarab and Olga Mordova; Karel was a witness."),
                         ["LEAK"] * 3)

    def test_stem_plus_ia_is_not_a_form_of_the_word(self) -> None:
        patterns = self.patterns(names="Швец Франц Хорват Хмурова")
        self.assertEqual(kinds(patterns, "Швеция\nФранция\nХорватия"), [])

    def test_names_that_end_like_a_name_form_plus_ia_are_candidates(self) -> None:
        patterns = self.patterns(names="Ксен Анастас Хмурова")
        self.assertEqual(kinds(patterns, "Ксен\nАнастас"), ["LEAK"] * 2)

    def test_short_ei_names_outside_name_fields_keep_case_forms(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "relatives_as_reported"],
                  [{"person_id": "P-9001", "relatives_as_reported": "Хмурова Анна (мать), Евсей (брат), Мокей"}])
        write_artifacts(self.case, summary="Записан крестник Авдей.")
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "Евсея\nЕвсеем\nМокея\nАвдея\nАвдеем"), ["LEAK"] * 5)

    def test_name_forms_of_a_rare_word_skip_forms_of_common_names(self) -> None:
        patterns = self.patterns(names="Ксений Хмурова")
        self.assertEqual(kinds(patterns, "Ксений"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Ксения\nКсении"), [])

    def test_extra_deny_listed_name_does_not_catch_other_names(self) -> None:
        patterns = self.patterns(extra_deny=["Александра", "Валентина", "Виктор", "Герман"])
        self.assertEqual(kinds(patterns, "Александр\nВалентин\nВиктория\nГермания"), [])
        self.assertEqual(kinds(patterns, "Александры\nВикторы\nГерманов"), ["LEAK"] * 3)
        self.assertEqual(kinds(self.patterns(extra_deny=["Евгения"]), "Евгений\nЕвгении"), ["LEAK"])
        # «Валентины» — и множественное «Валентин», и родительный «Валентина»: форма другого имени не ловится.
        self.assertEqual(kinds(self.patterns(extra_deny=["Валентин"]), "Валентины\nВалентинам"), ["LEAK"])

    def test_short_oi_name_taken_for_an_adjective_keeps_noun_forms(self) -> None:
        patterns = self.patterns(names="Сысой Хмурова")
        self.assertEqual(kinds(patterns, "Сысой\nСысоя\nСысою\nСысоем"), ["LEAK"] * 4)

    def test_short_ii_names_taken_for_adjectives_keep_noun_forms(self) -> None:
        patterns = self.patterns(names="Мокий Ипатий Акакий Хмурова")
        self.assertEqual(kinds(patterns, "Мокия\nМокию\nМокием\nИпатия\nАкакия"), ["LEAK"] * 5)

    def test_allow_in_an_oblique_form_covers_a_short_ei_name(self) -> None:
        patterns = self.patterns(names="Ерофей Хмурова", allow=["Ерофея"])
        self.assertEqual(kinds(patterns, "Ерофей\nЕрофея\nЕрофеем"), [])

    def test_extra_deny_listed_name_keeps_plural_forms(self) -> None:
        patterns = self.patterns(extra_deny=["Олег"])
        self.assertEqual(kinds(patterns, "Олеги\nОлегам\nОлегах"), ["LEAK"] * 3)

    def test_adjective_oblique_place_does_not_get_noun_forms(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "places"],
                  [{"person_id": "P-9001", "places": "с. Зимней Ставки"}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "Зимней"), ["LEAK"])
        self.assertEqual(kinds(patterns, "зимнее утро"), [])

    def test_hyphenated_case_form_in_skill_text_goes_to_review(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "places"],
                  [{"person_id": "P-9001", "places": "Ново-Сысоль"}])
        patterns = build_patterns([self.case], reference_text="Путь в Ново-Сысоли.")
        self.assertNotIn("LEAK", kinds(patterns, "в Ново-Сысоли"))

    def test_lowercase_allow_drops_only_that_form(self) -> None:
        patterns = self.patterns(names="Коваль Хмурова", allow=["ковали"])
        self.assertEqual(kinds(patterns, "мастера ковали железо"), [])
        self.assertEqual(kinds(patterns, "Коваль\nКоваля"), ["LEAK"] * 2)

    def test_extra_deny_common_name_uses_exact_forms_not_a_short_stem(self) -> None:
        patterns = self.patterns(extra_deny=["Лев"])
        self.assertEqual(kinds(patterns, "Лев\nЛьва\nЛьвом\nЛьвович"), ["LEAK"] * 4)
        self.assertEqual(kinds(patterns, "левый берег\nс левой стороны\nЛевая"), [])

    def test_patronymics_of_variant_names_follow_their_standard_form(self) -> None:
        patterns = self.patterns(extra_deny=["Петро", "Никола", "Илия"])
        self.assertEqual(kinds(patterns, "Петрович\nПетровна\nНиколаевич\nНиколаевна\nИльич\nИльинична"),
                         ["LEAK"] * 6)
        self.assertEqual(kinds(patterns, "Петроович\nИлиич"), [])

    def test_allow_in_an_oblique_form_covers_the_base_form(self) -> None:
        patterns = self.patterns(names="Анатолий Коваль Хмурова", allow=["Анатолия", "Коваля"])
        self.assertEqual(kinds(patterns, "Анатолий\nКоваль"), [])
        self.assertEqual(kinds(patterns, "Хмурова"), ["LEAK"])

    def test_allow_that_is_also_a_form_of_another_name_word_is_reported(self) -> None:
        patterns = self.patterns(names="Валерий Хмурова", allow=["Валерия"])
        self.assertEqual(kinds(patterns, "Валерий"), [])
        self.assertEqual(len(patterns.notes), 1, patterns.notes)
        self.assertIn("«Валерия»", patterns.notes[0])
        self.assertIn("«Валерий»", patterns.notes[0])
        self.assertEqual(self.patterns(names="Валерий Хмурова", allow=["Валерий"]).notes, ())
        free_text = self.patterns(names="Хмурова", summary="Свидетель — сосед Валерий.", allow=["Валерия"])
        self.assertIn("«Валерий»", " ".join(free_text.notes))

    def test_extra_deny_code_matches_across_scripts(self) -> None:
        patterns = self.patterns(extra_deny=["Р-1907"])
        self.assertEqual(kinds(patterns, "Р-1907\nP-1907"), ["LEAK"] * 2)
        self.assertEqual(kinds(patterns, "Р-19071"), [])

    def test_cyrillic_letter_variants_are_normalized(self) -> None:
        patterns = self.patterns(names="Коваль")
        self.assertEqual(kinds(patterns, "Коᲀаль\nКᲂваль"), ["LEAK"] * 2)

    def test_any_non_common_form_keeps_shared_pattern_leak(self) -> None:
        patterns = self.patterns(summary="Дом у Ольшанки, а ольшанки рядом; жили в Ольшанке.")
        self.assertEqual(kinds(patterns, "Ольшанка"), ["LEAK"])

    def test_next_ids_of_the_case_go_to_review(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "notes"],
                  [{"person_id": "P-9001", "notes": ""}, {"person_id": "P-9003", "notes": ""}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "см. P-9001."), ["LEAK"])
        self.assertEqual(kinds(patterns, "см. P-9004 и P-9053."), ["REVIEW"] * 2)
        self.assertEqual(kinds(patterns, "P-9002 P-9054 P-90041 AP-9004 P-904"), [])

    def test_ids_seen_anywhere_in_the_skill_go_to_review(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "names_as_written"],
                  [{"person_id": "P-9001", "names_as_written": "Тихомиров"}])
        patterns = build_patterns([self.case], reference_text=Reference("", id_text='ID = "P-9001"  # Тихомиров'))
        self.assertEqual(kinds(patterns, "P-9001"), ["REVIEW"])
        self.assertEqual(kinds(patterns, "Тихомиров"), ["LEAK"])

    def test_extra_deny_catches_common_looking_name(self) -> None:
        self.assertEqual(kinds(self.patterns(names="Варвара"), "крестница Варвара"), [])
        patterns = self.patterns(names="Варвара", extra_deny=["Варвара"], reference_text="Варвара")
        self.assertEqual(kinds(patterns, "крестница Варвара"), ["LEAK"])

    def test_extra_deny_common_name_keeps_every_form_and_its_patronymics(self) -> None:
        names = "Фома Фомич Фоминична Пётр Петрович Петровна Хмурова"
        common = self.patterns(names=names)
        self.assertEqual(kinds(common, "Фомы\nФоминичной\nПетра\nПетровной"), [])
        patterns = self.patterns(names=names, extra_deny=["Фома", "ПЁТР", "Вера"])
        text = "Фома\nФомы\nФомой\nФомич\nФомича\nФоминична\nФоминичной\nПётр\nПетра\nПетром\nПетрович\nПетровича\nПетровна\nПетровной"
        self.assertEqual(kinds(patterns, text), ["LEAK"] * 14)
        self.assertEqual(kinds(patterns, "Petrovna\nFominichna\nВеры\nВерой"), ["LEAK"] * 4)
        self.assertEqual(kinds(patterns, "Фомин\nФоменко\nв верности"), [])
        surname = self.patterns(names=names, extra_deny=["Мельник"])
        self.assertEqual(kinds(surname, "Мельника\nМельникович"), ["LEAK"])

    def test_extra_deny_of_one_case_wins_after_merging_cases(self) -> None:
        names = "Фома Фоминична Хмурова"
        merged = merge_patterns([self.patterns(names=names), self.patterns(names=names, extra_deny=["Фома"])])
        self.assertEqual(kinds(merged, "Фомы\nФоминичной"), ["LEAK"] * 2)

    def test_extra_deny_without_letters_is_an_exact_token(self) -> None:
        patterns = self.patterns(extra_deny=["1897"])
        self.assertEqual(kinds(patterns, "в 1897 году"), ["LEAK"])
        self.assertEqual(kinds(patterns, "11897 18970"), [])

    def test_sentence_start_is_skipped_and_register_lowercase_goes_to_review(self) -> None:
        patterns = self.patterns(summary="Кривоборье упомянуто. Дом у Ольшанки, а ольшанки рядом.")
        self.assertEqual(kinds(patterns, "Кривоборье"), [])
        self.assertEqual(kinds(patterns, "Ольшанки"), ["REVIEW"])
        self.assertEqual(kinds(self.patterns(summary="Жили в Кривоборье."), "Кривоборье"), ["LEAK"])

    def test_words_after_brackets_quotes_and_abbreviations_are_candidates(self) -> None:
        patterns = self.patterns(summary="Мать урожд. Ракитина, крест. Дубняков (Ольшанки), колхоз «Каменка»; Зябликово")
        for word in ("Ракитина", "Дубняков", "Ольшанки", "Каменка", "Зябликово"):
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])

    def test_name_and_place_fields_keep_first_word(self) -> None:
        write_csv(self.case / "person-register.csv",
                  ["person_id", "preferred_label", "normalized_name_variants", "places"],
                  [{"person_id": "P-9001", "preferred_label": "Тихомиров Иван",
                    "normalized_name_variants": "Залесский; Залесская", "places": "Кривоборье, губ."}])
        patterns = build_patterns([self.case])
        for word in ("Тихомиров", "Залесский", "Кривоборье"):
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])

    def test_capitals_ukrainian_letters_and_apostrophes(self) -> None:
        patterns = self.patterns(names="Єлисеєнко Ґонтарук Сем'янчук",
                                 summary="Записан ТИХОМИРОВ Иван; запрос в ЦАМО Министерства обороны, ф. 58")
        for word in ("Тихомиров", "Єлисеєнко", "Ґонтарук", "Сем'янчук"):
            with self.subTest(word):
                self.assertEqual(kinds(patterns, word), ["LEAK"])
        self.assertEqual(kinds(patterns, "ЦАМО"), [])

    def test_words_lowercase_in_reference_text_are_common(self) -> None:
        patterns = self.patterns(summary="Ответ: «Книгу не нашли», запрос в «Каменку»",
                                 reference_text="Прочитайте книгу целиком.")
        self.assertEqual(kinds(patterns, "книгу"), [])
        self.assertEqual(kinds(patterns, "Каменка"), ["LEAK"])

    def test_reference_text_words_go_to_review(self) -> None:
        patterns = self.patterns(names="Каменка Гончарова Тихомиров",
                                 reference_text="Пример: село Каменка, семья Гончаровых.")
        self.assertEqual(kinds(patterns, "Каменка"), ["REVIEW"])
        self.assertEqual(kinds(patterns, "Гончарова"), ["REVIEW"])
        self.assertEqual(kinds(patterns, "Тихомиров"), ["LEAK"])

    def test_allow_removes_word_in_any_form(self) -> None:
        self.assertEqual(kinds(self.patterns(names="Тихомиров", allow=["Тихомиров"]), "Тихомиров"), [])
        patterns = self.patterns(names="Тихомирова", allow=["Тихомиров"])
        self.assertEqual(kinds(patterns, "Тихомиров, Тихомировых"), [])

    def test_register_ids_are_exact_patterns(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "notes"], [{"person_id": "P-9001", "notes": ""}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "см. P-9001."), ["LEAK"])
        self.assertEqual(kinds(patterns, "P-90011 AP-9001"), [])
        self.assertEqual(kinds(build_patterns([self.case], reference_text="пример P-9001"), "P-9001"), ["REVIEW"])

    def test_register_ids_are_stripped_and_blank_ids_skipped(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "notes"],
                  [{"person_id": " P-9002 ", "notes": ""}, {"person_id": "   ", "notes": ""}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "P-9002 в начале"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Line.  "), [])

    def test_id_with_yo_is_exact(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "notes"], [{"person_id": "Ё-9001", "notes": ""}])
        patterns = build_patterns([self.case], reference_text="см. Е-9001")
        self.assertEqual(kinds(patterns, "Ё-9001"), ["LEAK"])

    def test_extra_csv_field_is_an_error(self) -> None:
        (self.case / "artifact-register.csv").write_text(
            "artifact_id,names_as_written\nA-9001,Тихомиров,лишнее поле\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_patterns([self.case])

    def test_diff_checks_added_lines_and_new_paths(self) -> None:
        patterns = self.patterns(names="Тихомиров Каменка")
        diff = ("diff --git a/x.md b/Тихомиров.md\n--- a/x.md\n+++ b/Тихомиров.md\n"
                "@@ -1,2 +1,3 @@\n-Тихомиров\n Тихомиров\n+семья Тихомиров\n+++ Каменка\n"
                "rename to Каменка.md")
        hits = scan_lines(diff.split("\n"), patterns, diff=True)
        self.assertEqual([(hit.kind, hit.line_no, hit.text) for hit in hits],
                         [("LEAK", 3, "b/Тихомиров.md"), ("LEAK", 7, "семья Тихомиров"),
                          ("LEAK", 8, "++ Каменка"), ("LEAK", 9, "Каменка.md")])
        self.assertEqual(len(scan_lines(diff.split("\n"), patterns)), 7)

    def test_diff_checks_path_of_new_empty_file(self) -> None:
        patterns = self.patterns(names="Тихомирова")
        diff = "diff --git a/Тихомирова.md b/Тихомирова.md\nnew file mode 100644\nindex 0000000..e69de29"
        self.assertEqual(kinds(patterns, diff, diff=True), ["LEAK"])

    def test_diff_skips_old_paths_of_deleted_and_renamed_files(self) -> None:
        patterns = self.patterns(names="Хмуров")
        diff = ("diff --git a/Хмуров.md b/Хмуров.md\ndeleted file mode 100644\n--- a/Хмуров.md\n+++ /dev/null\n"
                "@@ -1 +0,0 @@\n-x\n"
                "diff --git a/Хмурову.md b/x.md\nsimilarity index 100%\nrename from Хмурову.md\nrename to x.md\n"
                "diff --git a/Хмуров.png b/Хмуров.png\ndeleted file mode 100644\n"
                "Binary files a/Хмуров.png and /dev/null differ")
        self.assertEqual(kinds(patterns, diff, diff=True), [])
        added = ("diff --git a/Хмуров.png b/Хмуров.png\nnew file mode 100644\n"
                 "Binary files /dev/null and b/Хмуров.png differ")
        self.assertEqual(kinds(patterns, added, diff=True), ["LEAK"])

    def test_diff_reads_quoted_paths(self) -> None:
        patterns = self.patterns(names="Хмуров")
        quoted = '"b/' + "".join(f"\\{byte:03o}" for byte in "Хмуров.md".encode()) + '"'
        self.assertEqual(kinds(patterns, f"--- a/x.md\n+++ {quoted}", diff=True), ["LEAK"])
        self.assertEqual(kinds(patterns, f"diff --git {quoted.replace('b/', 'a/')} {quoted}\n"
                                         "new file mode 100644", diff=True), ["LEAK"])

    def test_diff_without_hunk_headers_checks_plus_lines(self) -> None:
        patterns = self.patterns(names="Тихомиров")
        self.assertEqual(kinds(patterns, "+семья Тихомиров\n-Тихомиров\n Тихомиров", diff=True), ["LEAK"])

    def test_short_possessive_surnames_match_all_forms(self) -> None:
        patterns = self.patterns(names="Шуева Бединой")
        self.assertEqual(kinds(patterns, "Шуев\nШуевой\nШуеву\nШуевых\nБедин\nБедина\nБединых"), ["LEAK"] * 7)
        self.assertEqual(kinds(self.patterns(names="Шуевой"), "Шуев\nШуева"), ["LEAK"] * 2)

    def test_surnames_with_fleeting_vowel_match_oblique_forms(self) -> None:
        patterns = self.patterns(names="Травец Хорел Буровей")
        self.assertEqual(kinds(patterns, "Травца\nТравцом\nХорла\nБуровья\nБуровьем"), ["LEAK"] * 5)
        self.assertEqual(kinds(self.patterns(names="Травца"), "Травец"), ["LEAK"])

    def test_linking_part_of_hyphenated_place_gets_exact_pattern(self) -> None:
        patterns = self.patterns(names="Хмурово-Каменка")
        self.assertEqual(kinds(patterns, "Хмурово\nКаменка"), ["LEAK"] * 2)
        self.assertEqual(kinds(patterns, "Хмуров\nХмуровы"), [])

    def test_every_dropped_name_field_word_is_reported_until_decided(self) -> None:
        patterns = self.patterns(names="Мельник Ключи", reference_text="один мельник; ключи")
        self.assertEqual(kinds(patterns, "Мельник\nКлючи"), [])
        self.assertEqual(len(patterns.notes), 1, patterns.notes)
        self.assertIn("«Ключи», «Мельник»", patterns.notes[0])
        patterns = self.patterns(names="Мельник Ключи", reference_text="один мельник; ключи",
                                 allow=["ключи"], extra_deny=["Мельник"])
        self.assertEqual(patterns.notes, ())
        self.assertEqual(kinds(patterns, "Мельником"), ["LEAK"])

    def test_list_fields_take_every_capitalized_word(self) -> None:
        write_csv(self.case / "person-register.csv", ["person_id", "relatives_as_reported"],
                  [{"person_id": "P-9001", "relatives_as_reported": "Гончарова Мария (мать). Ракитин Пётр (брат)."}])
        self.assertEqual(kinds(build_patterns([self.case]), "Гончарова\nРакитин"), ["LEAK"] * 2)

    def test_more_common_given_names_and_patronymics_are_not_candidates(self) -> None:
        common = "Надежда Роман Вадим Фома Романовна Юрьевна Льва"
        male = "Ильич Фомич Данилович Вадимович Львович"
        patterns = self.patterns(names=f"{common} {male} Хмурова")
        self.assertEqual(kinds(patterns, "\n".join(common.split())), [])
        self.assertEqual(kinds(patterns, "\n".join(male.split())), ["REVIEW"] * 5)
        self.assertEqual(kinds(patterns, "Хмурова"), ["LEAK"])

    def test_record_numbers_from_register_fields_are_patterns(self) -> None:
        write_csv(self.case / "artifact-register.csv",
                  ["artifact_id", "archival_reference", "source_url_or_path", "sha256"],
                  [{"artifact_id": "A-9001", "archival_reference": "ф. Р-9907, оп. 2, д. 678, 1-я часть",
                    "source_url_or_path": "https://www.familysearch.org/ark:/61903/3:1:3QS7-L9QQ-ZZZZ",
                    "sha256": "a" * 64}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "см. Р-9907\n3QS7-L9QQ-ZZZZ\n" + "a" * 64), ["LEAK"] * 3)
        self.assertEqual(kinds(patterns, "оп. 2, д. 678; 61903; 1-я часть"), [])

    def test_latin_spelling_of_case_words_is_leak(self) -> None:
        patterns = self.patterns(names="Тихомирова Каменка Ранов")
        self.assertEqual(kinds(patterns, "Add the Tikhomirov example\nTihomirovs of Kamenka\nranova"),
                         ["LEAK"] * 4)
        self.assertEqual(kinds(patterns, "beloved Ranovsky ranch"), [])

    def test_hidden_characters_and_latin_lookalikes_do_not_break_words(self) -> None:
        patterns = self.patterns(names="Тихомиров")
        self.assertEqual(kinds(patterns, "Тихо\u200bмиров\nТихо\u2060миров\nТихомирoв"), ["LEAK"] * 3)

    def test_case_insensitive_index_keeps_letter_positions(self) -> None:
        patterns = self.patterns(extra_deny=["Tikhomirov", "ßeta"])
        self.assertEqual(kinds(patterns, "TİKHOMIROV\nßeta"), ["LEAK"] * 2)

    def test_latin_extra_deny_with_diacritics_matches_word_beginnings(self) -> None:
        patterns = self.patterns(extra_deny=["Wiśniewski"])
        self.assertEqual(kinds(patterns, "Wiśniewskiego"), ["LEAK"])

    def test_diff_reads_new_file_paths_with_other_prefixes(self) -> None:
        patterns = self.patterns(names="Хмуров")
        for header in ("diff --git Хмуров.md Хмуров.md", "diff --git c/Хмуров.md w/Хмуров.md",
                       "diff --git c/x.md w/Хмуров.md", "diff --git Хмуров notes.md Хмуров notes.md"):
            with self.subTest(header=header):
                self.assertEqual(kinds(patterns, header + "\nnew file mode 100644", diff=True), ["LEAK"])

    def test_male_patronymic_forms_go_to_review_female_are_dropped(self) -> None:
        patterns = self.patterns(names="Вадимович Вадимовна Хмурова")
        self.assertEqual(kinds(patterns, "Вадимович\nВадимовича"), ["REVIEW"] * 2)
        self.assertEqual(kinds(patterns, "Вадимовна"), [])

    def test_given_names_are_recognized_by_their_forms_not_by_a_shared_start(self) -> None:
        patterns = self.patterns(names="Андра Христин Андрея Христиной")
        self.assertEqual(kinds(patterns, "Андрея\nХристиной"), [])
        self.assertEqual(kinds(patterns, "Андра"), ["LEAK"])
        self.assertEqual(len(patterns.notes), 1, patterns.notes)
        self.assertIn("«Христин»", patterns.notes[0])
        patterns = self.patterns(names="Павл Хмурова")
        self.assertEqual(kinds(patterns, "Павле"), [])
        self.assertIn("«Павл»", patterns.notes[0])
        self.assertNotIn("Андра", patterns.notes[0])

    def test_words_written_in_caps_by_reference_are_reported(self) -> None:
        patterns = self.patterns(names="Хмуров", reference_text="ХМУРОВ Иван")
        self.assertEqual(kinds(patterns, "Хмуров"), [])
        self.assertEqual(len(patterns.notes), 1, patterns.notes)
        self.assertIn("«Хмуров»", patterns.notes[0])

    def test_latin_name_with_cyrillic_ending_keeps_latin_letters(self) -> None:
        patterns = self.patterns(extra_deny=["Kowalczyk"])
        self.assertEqual(kinds(patterns, "Kowalczykом"), ["LEAK"])

    def test_short_adjective_surname_sharing_a_start_with_a_name(self) -> None:
        patterns = self.patterns(names="Тихая")
        self.assertEqual(kinds(patterns, "Тихий\nТихая\nТихого"), ["LEAK"] * 3)

    def test_linking_part_equal_to_given_name_is_not_a_pattern(self) -> None:
        patterns = self.patterns(names="Петро-Хмуровка")
        self.assertEqual(kinds(patterns, "Петро"), [])
        self.assertEqual(kinds(patterns, "Хмуровка"), ["LEAK"])

    def test_allow_of_one_form_covers_fleeting_vowel_forms(self) -> None:
        patterns = self.patterns(names="Травец Травца", allow=["Травец"])
        self.assertEqual(kinds(patterns, "Травец\nТравца\nТравцом"), [])

    def test_latin_spellings_with_diacritics_x_and_off(self) -> None:
        patterns = self.patterns(names="Плексинов Чернавский Тихомиров")
        self.assertEqual(kinds(patterns, "Plexinov\nČernavskij\nTikhomiroff"), ["LEAK"] * 3)

    def test_latin_key_of_a_word_the_skill_writes_in_latin_is_dropped(self) -> None:
        patterns = self.patterns(summary="Писали через Телеграм каждый день.", reference_text="Канал в Telegram.")
        self.assertEqual(kinds(patterns, "the Telegram channel"), [])
        self.assertFalse(any("-" in key for key in self.patterns(names="Хмурово-Каменка").latin))

    def test_more_invisible_characters_are_removed_before_normalization(self) -> None:
        patterns = self.patterns(names="Хмуровская Клёнова")
        self.assertEqual(kinds(patterns, "Хмуро\u034fвская\nХмуро\ufe0fвская\nКле\u200b\u0308нова"), ["LEAK"] * 3)

    def test_dotless_i_keeps_fast_matcher_equal_to_regex(self) -> None:
        patterns = self.patterns(extra_deny=["Tikhomirov"])
        self.assertEqual(kinds(patterns, "Tıkhomirov"), ["LEAK"])

    def test_reference_codes_match_across_scripts_and_skip_ordinals(self) -> None:
        write_csv(self.case / "artifact-register.csv", ["artifact_id", "archival_reference", "source_url_or_path"],
                  [{"artifact_id": "A-9001", "archival_reference": "ф. Р-12; 125-й, 1905-го",
                    "source_url_or_path": "scans/image-001.jpg"}])
        patterns = build_patterns([self.case])
        self.assertEqual(kinds(patterns, "Р-12\nP-12"), ["LEAK"] * 2)
        self.assertEqual(kinds(patterns, "125-й 1905-го image-001"), [])

    def test_min_stem_is_five(self) -> None:
        self.assertEqual(MIN_STEM, 5)
        patterns = self.patterns(names="Сомова Ивина")
        self.assertEqual(kinds(patterns, "Сомов"), ["LEAK"])
        self.assertEqual(kinds(patterns, "Ивин\nИвина"), ["LEAK"] * 2)

    def test_case_without_registers_gives_no_patterns(self) -> None:
        self.assertEqual(build_patterns([self.case]), PatternSet((), ()))


class ReferenceAgreesWithRegexTest(unittest.TestCase):
    """Быстрый поиск в тексте навыка должен решать так же, как сам шаблон."""

    LETTERS = "аеёклнорыхмъуюйЕЁК"
    OTHER = " -.'1\n"

    def random_text(self, rng: random.Random) -> str:
        return "".join(rng.choice(self.LETTERS + self.OTHER) for _ in range(rng.randint(0, 60)))

    def random_word(self, rng: random.Random) -> str:
        tail = "".join(rng.choice(self.LETTERS.lower()) for _ in range(rng.randint(3, 7)))
        return rng.choice("КЕЁ") + tail + rng.choice(["", "", "-" + tail])

    def test_word_lookup_matches_regex(self) -> None:
        rng = random.Random(7)
        for _ in range(400):
            text = self.random_text(rng)
            reference = Reference(text)
            for _ in range(10):
                word = self.random_word(rng)
                expected = re.search(to_pattern(word), modern(text), re.IGNORECASE) is not None
                self.assertEqual(reference.has_word(word), expected, (word, text))

    def test_adjective_lookup_matches_regex(self) -> None:
        rng = random.Random(17)
        for _ in range(400):
            text = self.random_text(rng)
            reference = Reference(text)
            for _ in range(10):
                stem = rng.choice("КЕЁ") + "".join(rng.choice(self.LETTERS.lower()) for _ in range(2))
                expected = re.search(adjective_pattern(stem), modern(text), re.IGNORECASE) is not None
                self.assertEqual(reference.has_adjective(stem), expected, (stem, text))

    def test_fast_scan_matches_full_regex_scan(self) -> None:
        rng = random.Random(13)
        for _ in range(200):
            sources = [to_pattern(self.random_word(rng)) for _ in range(rng.randint(1, 6))]
            sources += [rf"(?<![\w-]){re.escape(value)}(?![\w-])" for value in ("Е-1", "Ё-1", "к-1", "ке")]
            sources += [r"к\w+", "Е"]
            patterns = PatternSet(tuple(sources[::2]), tuple(sources[1::2]))
            lines = [self.random_text(rng) for _ in range(5)]
            expected = [(kind, number, source)
                        for number, line in enumerate(lines, 1)
                        for kind, group in (("LEAK", patterns.deny), ("REVIEW", patterns.review))
                        for source in group if re.search(source, modern(line), re.IGNORECASE)]
            matcher = Matcher(patterns)
            found = [(hit.kind, hit.line_no, hit.pattern)
                     for number, line in enumerate(lines, 1) for hit in matcher.hits(number, modern(line))]
            self.assertEqual(sorted(found), sorted(expected), lines)

    def test_id_lookup_matches_regex(self) -> None:
        rng = random.Random(11)
        for _ in range(400):
            text = self.random_text(rng)
            reference = Reference(text)
            for value in ("Е-1", "Ё-1", "к-1", "о.1", "ке"):
                expected = re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", modern(text), re.IGNORECASE)
                self.assertEqual(reference.has_id(value), expected is not None, (value, text))


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_runs_and_fast_paths_agree(self) -> None:
        result = bench_privacy_scan.run(words=40, lines=40, seed=3)
        self.assertEqual(set(result), {"build_fast", "build_regex", "check_fast", "check_regex"})
        self.assertTrue(all(seconds >= 0 for seconds in result.values()))


class CommandLineTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.case = self.root / "case"
        self.case.mkdir()
        write_artifacts(self.case, names="Тихомиров Каменка")
        (self.case / "privacy-scan.json").write_text(json.dumps({"extra_deny": ["Марфа"], "allow": []}),
                                                     encoding="utf-8")
        self.reference = self.root / "skill"
        self.reference.mkdir()
        (self.reference / "guide.md").write_text("Пример: село Каменка.", encoding="utf-8")
        (self.reference / "SKILL.md").write_text("# Навык\n", encoding="utf-8")
        self.out = self.root / "out" / "patterns.json"

    def run_cli(self, *args: str, stdin: str | None = None,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPTS / "privacy_scan.py"), *args],
                              capture_output=True, text=True, input=stdin, env=env)

    def build(self, *case_dirs: Path, out: Path | None = None,
              env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        args = [value for case_dir in case_dirs or (self.case,) for value in ("--case-dir", str(case_dir))]
        return self.run_cli("build", *args, "--reference-dir", str(self.reference), "--out", str(out or self.out),
                            env=env)

    def check(self, text: str, *extra: str) -> subprocess.CompletedProcess[str]:
        path = self.root / "outgoing.txt"
        path.write_text(text, encoding="utf-8")
        return self.run_cli("check", "--patterns", str(self.out), *extra, str(path))

    def assert_error(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertTrue(result.stderr.startswith("Ошибка:"), result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_build_then_check(self) -> None:
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("deny:", result.stdout)
        leak = self.check("Тихомиров и Марфа")
        self.assertEqual(leak.returncode, 1, leak.stderr)
        self.assertEqual(leak.stdout.count("LEAK"), 2)
        review = self.check("село Каменка")
        self.assertEqual(review.returncode, 0, review.stderr)
        self.assertIn("REVIEW", review.stdout)
        self.assertEqual(self.check("ничего лишнего").returncode, 0)

    def test_cli_build_reads_every_case_dir(self) -> None:
        other = self.root / "other"
        other.mkdir()
        write_artifacts(other, names="Залесский")
        self.assertEqual(self.build(self.case, other).returncode, 0)
        self.assertEqual(self.check("Тихомиров, Залесский").stdout.count("LEAK"), 2)

    def test_cli_check_reads_diff_from_stdin(self) -> None:
        self.assertEqual(self.build().returncode, 0)
        added = self.run_cli("check", "--patterns", str(self.out), "--diff", "-", stdin="+++ b/x\n+Тихомиров\n")
        self.assertEqual(added.returncode, 1, added.stderr)
        removed = self.run_cli("check", "--patterns", str(self.out), "--diff",
                               stdin="--- a/x\n+++ b/x\n@@ -1 +0,0 @@\n-Тихомиров\n")
        self.assertEqual(removed.returncode, 0, removed.stderr)

    def test_cli_diff_mode_refuses_input_without_diff_structure(self) -> None:
        self.assertEqual(self.build().returncode, 0)
        colored = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n\x1b[32m+Тихомиров\x1b[m\n"
        plain = "Тихомиров в тексте\n- пункт списка\n"
        for label, text in (("colored", colored), ("plain text", plain)):
            with self.subTest(label):
                self.assert_error(self.run_cli("check", "--patterns", str(self.out), "--diff", stdin=text))
        for extra in (("--diff",), ()):
            with self.subTest(extra=extra):
                self.assert_error(self.run_cli("check", "--patterns", str(self.out), *extra, stdin=""))

    def test_cli_reference_reads_only_docs(self) -> None:
        (self.reference / "helper.py").write_text('EXAMPLE = "Тихомиров"\n', encoding="utf-8")
        (self.reference / "tests").mkdir()
        (self.reference / "tests" / "cases.md").write_text("Тихомиров\n", encoding="utf-8")
        self.assertEqual(self.build().returncode, 0)
        self.assertEqual(self.check("Тихомиров").returncode, 1)

    def test_cli_reference_must_be_a_skill_folder_without_cases(self) -> None:
        for label, reference in (("missing", self.root / "none"), ("file", self.reference / "guide.md"),
                                 ("no SKILL.md", self.case)):
            with self.subTest(label):
                self.assert_error(self.run_cli("build", "--case-dir", str(self.case), "--reference-dir", str(reference),
                                               "--out", str(self.out)))
        inside = self.reference / "my-case"
        inside.mkdir()
        write_artifacts(inside, names="Залесский")
        self.assert_error(self.build(inside))

    def test_cli_reference_skips_case_folders_and_untracked_files(self) -> None:
        stray = self.reference / "cases" / "old"
        stray.mkdir(parents=True)
        write_artifacts(stray, names="Тихомиров")
        (stray / "research-state.md").write_text("Тихомиров, P-9001\n", encoding="utf-8")
        (self.reference / "notes").mkdir()
        (self.reference / "notes" / "handoff.md").write_text("Тихомиров\n", encoding="utf-8")
        (self.reference / "references").mkdir()
        (self.reference / "references" / "guide2.md").write_text("Залесский\n", encoding="utf-8")
        write_artifacts(self.case, names="Тихомиров Каменка Залесский")
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(self.root)}
        self.assertEqual(self.build(env=env).returncode, 0)
        self.assertEqual(self.check("Тихомиров").returncode, 1)
        self.assertIn("REVIEW", self.check("Залесский").stdout)
        subprocess.run(["git", "init", "-q", str(self.reference)], check=True)
        subprocess.run(["git", "-C", str(self.reference), "add", "guide.md", "SKILL.md"], check=True)
        (self.reference / "notes.md").write_text("Тихомиров\n", encoding="utf-8")
        self.assertEqual(self.build().returncode, 0)
        self.assertEqual(self.check("Тихомиров").returncode, 1)
        self.assertIn("REVIEW", self.check("село Каменка").stdout)

    def test_cli_reference_partly_tracked_by_another_repo_is_read_as_folder(self) -> None:
        (self.reference / "references").mkdir()
        (self.reference / "references" / "guide2.md").write_text("Залесский\n", encoding="utf-8")
        write_artifacts(self.case, names="Тихомиров Залесский")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "skill/guide.md"], check=True)
        self.assertEqual(self.build().returncode, 0)
        self.assertIn("REVIEW", self.check("Залесский").stdout)

    def test_cli_allow_applies_only_to_its_case(self) -> None:
        other = self.root / "other"
        other.mkdir()
        write_artifacts(other, names="Залесский Тихомиров")
        (other / "privacy-scan.json").write_text(json.dumps({"allow": ["Тихомиров"]}), encoding="utf-8")
        result = self.build(self.case, other)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.check("Тихомиров").returncode, 1)
        self.assertEqual(self.check("Залесский").returncode, 1)

    def test_cli_build_prints_allow_notes(self) -> None:
        write_artifacts(self.case, names="Белая Тихомиров")
        (self.case / "privacy-scan.json").write_text(json.dumps({"allow": ["Белое"]}), encoding="utf-8")
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Белая", result.stderr)

    def test_cli_check_splits_lines_only_on_newline(self) -> None:
        self.assertEqual(self.build().returncode, 0)
        result = self.run_cli("check", "--patterns", str(self.out), "--diff", stdin="+см. Тихомиров\n")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("checked lines: 1;", result.stdout)

    def test_cli_build_refuses_output_inside_skill(self) -> None:
        target = SKILL_ROOT / "privacy-patterns.json"
        result = self.run_cli("build", "--case-dir", str(self.case), "--out", str(target))
        self.assert_error(result)
        self.assertFalse(target.exists())

    def test_cli_build_refuses_output_inside_reference_dir(self) -> None:
        target = self.reference / "patterns.json"
        self.assert_error(self.build(out=target))
        self.assertFalse(target.exists())

    def test_cli_build_errors(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "artifact-register.csv").write_text("artifact_id,summary\nA-1,x,y\n", encoding="utf-8")
        for label, case_dir in (("missing case dir", self.root / "none"), ("no registers", empty),
                                ("extra CSV field", broken)):
            with self.subTest(label):
                self.assert_error(self.build(case_dir))
                self.assertFalse(self.out.exists())

    def test_cli_config_errors(self) -> None:
        for label, config in (("unknown key", {"deny": ["Марфа"]}), ("blank entry", {"extra_deny": [" "]}),
                              ("not a string", {"allow": [1]})):
            with self.subTest(label):
                (self.case / "privacy-scan.json").write_text(json.dumps(config), encoding="utf-8")
                self.assert_error(self.build())

    def test_cli_check_errors(self) -> None:
        self.out.parent.mkdir()
        cases = (("missing patterns", None), ("not a list of strings", {"deny": [1], "review": []}),
                 ("invalid regex", {"deny": ["("], "review": []}))
        for label, data in cases:
            with self.subTest(label):
                if data is not None:
                    self.out.write_text(json.dumps(data), encoding="utf-8")
                self.assert_error(self.run_cli("check", "--patterns", str(self.out), stdin=""))


if __name__ == "__main__":
    unittest.main()
