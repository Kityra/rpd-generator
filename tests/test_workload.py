"""Этап 3: часы считает код по параметрам курса и config/workload_norms.yaml."""
import unittest

from domain.workload import (CLASS_HOURS, HOURS, WorkloadEdit, WorkloadParams, apply_workload_edit, assignment_kind,
                             load_norms, propose_workload, reported_workload_issues, subject_character, validate_workload,
                             workload)


def subject(key, title, topics=1, assignments=(), fundamental=False, importance=None):
    items = [{"id": f"{key}-t{index}", "title": f"{title}: тема {index}",
              "assignments": [{"id": f"{key}-a{index}-{n}", "title": text, "instructions": text, "expected_artifacts": []}
                              for n, text in enumerate(assignments if index == 0 else [])]}
             for index in range(topics)]
    return {"id": key, "title": title, "is_fundamental": fundamental, "importance": importance, "topics": items}


def params(**values):
    return WorkloadParams.from_meta({"workload": values})


def total(row):
    return sum(row[field] for field in HOURS)


class ParameterTests(unittest.TestCase):
    def test_two_courses_with_different_parameters_get_different_volumes(self):
        subjects = [subject("a", "Регрессия", 3), subject("b", "Статистика", 2)]
        bachelor = propose_workload(subjects, None, params(program_type="bachelor", total_credits=60, hours_per_credit=36))
        dpo = propose_workload(subjects, None, params(program_type="dpo", total_hours=144, hours_per_credit=36,
                                                      assessment_form="credit"))
        self.assertEqual(bachelor["total_hours"], 2160)
        self.assertEqual(dpo["total_hours"], 144)
        self.assertGreater(bachelor["loads"]["a"]["independent"] / total(bachelor["loads"]["a"]),
                           dpo["loads"]["a"]["independent"] / total(dpo["loads"]["a"]))
        self.assertEqual(dpo["loads"]["a"]["assessment"], 4)
        self.assertEqual(bachelor["loads"]["a"]["assessment"], 36)

    def test_old_course_keeps_previous_default(self):
        legacy = WorkloadParams.from_meta({"title": "Курс"})
        self.assertFalse(legacy.explicit)
        self.assertEqual((legacy.total_hours, legacy.hours_per_credit), (2160, 36))

    def test_volume_in_credits_or_hours(self):
        self.assertEqual(params(total_credits=10, hours_per_credit=27).total_hours, 270)
        self.assertEqual(params(total_hours=500, hours_per_credit=36).total_hours, 500)
        self.assertEqual(params(program_type="dpo").total_hours, 16 * 36)


class WeightTests(unittest.TestCase):
    def test_equal_weights_split_evenly_in_whole_credits(self):
        subjects = [subject(key, key, 2) for key in "abc"]
        proposal = propose_workload(subjects, None, params(total_credits=12, hours_per_credit=36))
        self.assertEqual([total(proposal["loads"][key]) for key in "abc"], [144, 144, 144])
        self.assertTrue(all(total(row) % 36 == 0 for row in proposal["loads"].values()))

    def test_topics_practical_tasks_core_and_importance_raise_weight(self):
        base = subject("base", "База", 1)
        cases = {"topics": subject("t", "Темы", 4), "practice": subject("p", "Практика", 1, ["Напишите SQL-запрос"]),
                 "core": subject("c", "Ядро", 1, fundamental=True), "rank": subject("r", "Ранг", 1, importance="core")}
        for name, other in cases.items():
            with self.subTest(name):
                proposal = propose_workload([base, other], None, params(total_credits=20, hours_per_credit=36))
                self.assertGreater(total(proposal["loads"][other["id"]]), total(proposal["loads"]["base"]))

    def test_labs_only_for_laboratory_tasks(self):
        norms = load_norms()
        theory = subject("a", "История науки", 2)
        practice = subject("b", "Анализ требований", 2, ["Проанализируйте требования заказчика и подготовьте отчёт"])
        lab = subject("c", "Машинное обучение", 2, ["Обучите модель на наборе данных и запустите эксперимент"])
        self.assertEqual([subject_character(item, norms) for item in (theory, practice, lab)],
                         ["theoretical", "practical", "laboratory"])
        self.assertIsNone(assignment_kind({"title": "Обсуждение", "instructions": "Обсудите понятия"}, norms))
        proposal = propose_workload([theory, practice, lab], None, params(total_credits=15, hours_per_credit=36))
        self.assertEqual(proposal["loads"]["a"]["labs"], 0)
        self.assertEqual(proposal["loads"]["b"]["labs"], 0)
        self.assertGreater(proposal["loads"]["c"]["labs"], 0)
        self.assertGreater(proposal["loads"]["b"]["practice"] / total(proposal["loads"]["b"]),
                           proposal["loads"]["a"]["practice"] / total(proposal["loads"]["a"]))


class AssessmentAndChecksTests(unittest.TestCase):
    def test_exam_and_credit_hours_follow_form(self):
        subjects = [subject("a", "Регрессия", 2)]
        exam = propose_workload(subjects, None, params(total_credits=6, assessment_form="exam"))
        credit = propose_workload(subjects, None, params(total_credits=6, assessment_form="credit"))
        graded = propose_workload(subjects, None, params(total_credits=6, assessment_form="graded_credit"))
        self.assertEqual([item["loads"]["a"]["assessment"] for item in (exam, credit, graded)], [36, 4, 6])
        for item in (exam, credit, graded):
            row = item["loads"]["a"]
            self.assertEqual(total(row), 216)
            self.assertEqual(sum(row["topics"].values()), sum(row[field] for field in CLASS_HOURS))
            self.assertEqual(validate_workload(subjects, item, params(total_credits=6)), [])

    def test_shortage_of_hours_is_reported(self):
        subjects = [subject(key, key) for key in "abc"]
        # 2 з.е. на три дисциплины: целыми з.е. не делится, дисциплины меньше экзамена.
        proposal = propose_workload(subjects, None, params(total_credits=2, hours_per_credit=36, assessment_form="exam"))
        self.assertEqual(proposal["total_hours"], 72)
        issues = validate_workload(subjects, proposal, params(total_credits=2))
        self.assertTrue(any("не кратно зачётной единице" in issue for issue in issues))
        self.assertTrue(any("не осталось часов" in issue for issue in issues))
        current = propose_workload(subjects, None, params(total_credits=6))
        with self.assertRaises(ValueError):
            apply_workload_edit(subjects, current, WorkloadEdit(
                workload_only=True, total_hours=None, total_credits=None, hours_per_credit=None, distribute_evenly=False,
                subjects=[{"subject_id": "a", "total_hours": 300, "credits": None, "lectures": None, "practice": None,
                           "labs": None, "independent": None, "assessment": None}]))

    def test_validation_catches_mismatches(self):
        subjects = [subject("a", "Регрессия", 2), subject("b", "Статистика", 1)]
        settings = params(total_credits=10, assessment_form="exam")
        proposal = propose_workload(subjects, None, settings)
        broken = {**proposal, "loads": {**proposal["loads"], "a": {**proposal["loads"]["a"], "lectures":
                                                                proposal["loads"]["a"]["lectures"] + 5, "assessment": 2}}}
        issues = validate_workload(subjects, broken, settings)
        text = "\n".join(issues)
        self.assertIn("не равна объёму", text)
        self.assertIn("на аттестацию", text)
        self.assertIn("не равна объёму курса", text)
        self.assertIn("не совпадает с объёмом курса из параметров", text)

    def test_semesters_put_core_first(self):
        subjects = [subject("a", "Прикладная", 1), subject("b", "Ядро", 1, fundamental=True), subject("c", "Ещё", 1)]
        proposal = propose_workload(subjects, None, params(total_credits=12, semesters=2))
        self.assertEqual(proposal["loads"]["b"]["semester"], 1)
        self.assertEqual({row["semester"] for row in proposal["loads"].values()}, {1, 2})


class EditTests(unittest.TestCase):
    def setUp(self):
        self.subjects = [subject("reg", "Регрессия", 3), subject("stat", "Статистика", 2), subject("viz", "Визуализация", 1)]
        self.params = params(total_credits=12, hours_per_credit=36, assessment_form="credit")
        self.current = propose_workload(self.subjects, None, self.params)

    def edit(self, **values):
        changes = values.pop("subjects", [])
        return WorkloadEdit(workload_only=True, total_hours=values.get("total_hours"), total_credits=None,
                            hours_per_credit=None, distribute_evenly=values.get("evenly", False), subjects=[
                                {"subject_id": key, "total_hours": hours, "credits": None, "lectures": None,
                                 "practice": None, "labs": None, "independent": None, "assessment": None}
                                for key, hours in changes])

    def test_give_regression_72_hours(self):
        result = apply_workload_edit(self.subjects, self.current, self.edit(subjects=[("reg", 72)]))
        self.assertEqual(total(result["loads"]["reg"]), 72)
        self.assertEqual(result["total_hours"], 432)
        self.assertEqual(total(result["loads"]["stat"]) + total(result["loads"]["viz"]), 360)
        self.assertEqual(result["loads"]["reg"]["assessment"], 4)
        self.assertEqual(sum(result["loads"]["reg"]["topics"].values()), 68)
        self.assertEqual(result["loads"]["reg"]["credits"], 2)

    def test_distribute_evenly_and_change_total(self):
        result = apply_workload_edit(self.subjects, self.current, self.edit(total_hours=324, evenly=True))
        self.assertEqual([total(result["loads"][key]) for key in ("reg", "stat", "viz")], [108, 108, 108])
        self.assertEqual(validate_workload(self.subjects, result), [])


class ReportedHoursTests(unittest.TestCase):
    def test_cli_hours_are_checked_arithmetically(self):
        good = {"totalHours": 144, "totalCredits": 4, "bySemester": [
            {"semester": 1, "hours": 144, "lectureHours": 36, "seminarHours": 36, "independentHours": 72}]}
        self.assertEqual(reported_workload_issues(good, 36), [])
        bad = {"totalHours": 150, "totalCredits": 4, "bySemester": [
            {"semester": 1, "hours": 100, "lectureHours": 36, "seminarHours": 36}, {"semester": 2, "hours": 40}]}
        issues = reported_workload_issues(bad, 36)
        self.assertEqual(len(issues), 3)


class LegacyRowsTests(unittest.TestCase):
    def test_archived_rows_without_new_fields_are_valid(self):
        row = {"lectures": 10, "practice": 10, "labs": 0, "independent": 16, "hours_per_credit": 36}
        self.assertEqual(workload(row), {"totalHours": 36, "totalCredits": 1})
        previous = {"loads": {"a": row}, "hours_per_credit": 36, "origin": "teacher", "total_hours": 36}
        self.assertIs(propose_workload([subject("a", "A")], previous), previous)


if __name__ == "__main__":
    unittest.main()
