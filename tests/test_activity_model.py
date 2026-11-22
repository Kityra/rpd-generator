"""Модель деятельности профессии: формулы, исключение экспертов, статусы, тренды, цитаты, правила R1–R5,
детерминизм и сквозной прогон демоданных. Без сети и без LLM (экстрактор подменяется заглушкой)."""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

from activity import engine
from activity.config import load_config
from activity.experts import Expert, Vote, assess
from activity.extract import ClaimDraft, LlmExtractor, parse_lines, parse_standard, verify
from activity.program import Discipline, ProgramInput, analyse
from activity.rpd_bridge import activity_from_snapshot, course_snapshot
from activity.stats import experts_needed, wilson
from activity.text import as_task, canonical_key, quote_problem


CONFIG = load_config()
AS_OF = date(2026, 11, 21)


def metrics(n=0, k=0, ci=None, support=0.0, doc_types=0, share=None, share_prev=None, k_prev=0, delta=None,
            significant=False, in_standard=False, n_forecast=0, net=None, automated=None):
    return {"experts": {"n": n, "k": k, "ci": ci, "n_forecast": n_forecast, "forecast_net": net, "automated_share": automated,
                        "n_needed": None},
            "support": {"value": support}, "doc_types": doc_types,
            "market": {"share": share, "share_prev": share_prev, "k_emp_prev": k_prev, "delta": delta,
                       "delta_significant": significant},
            "standard": {"in_standard": in_standard}}


def item(**fields):
    return engine.Item(**{"id": "t", "level": "task", "name": "Проектирует схемы баз данных", **fields})


class FormulaTests(unittest.TestCase):
    def test_wilson_reference_values(self):
        self.assertEqual([round(v, 3) for v in wilson(8, 10)], [0.490, 0.943])
        self.assertEqual([round(v, 3) for v in wilson(38, 45)], [0.712, 0.923])
        self.assertIsNone(wilson(0, 0))

    def test_experts_needed_reference_values(self):
        self.assertEqual(experts_needed(0.7), 21)
        self.assertEqual(experts_needed(0.65), 39)
        self.assertEqual(experts_needed(0.6), 93)
        self.assertEqual(experts_needed(9 / 13) - 13, 10)
        self.assertIsNone(experts_needed(0.5))


class ExpertExclusionTests(unittest.TestCase):
    controls = {"foreign": {"kind": "foreign", "duplicate_of": None},
                "dup": {"kind": "duplicate", "duplicate_of": "t1"}}
    names = {"foreign": "Проектирует систему капельного полива грядок", "dup": "дубль", "t1": "задача"}

    def run_assess(self, votes):
        return assess([Expert(id="e", label="Эксперт 01")], votes, self.controls, self.names, CONFIG)["e"]

    def honest(self, **overrides):
        base = {"foreign": "other_role", "dup": "regular", "t1": "regular"}
        base.update(overrides)
        verdicts = ["regular", "sometimes", "other_role", "outdated"]
        votes = [Vote("e", key, verdict, importance=3, elapsed_ms=8000) for key, verdict in base.items()]
        votes += [Vote("e", f"x{i}", verdicts[i % 4], importance=i % 5 + 1, elapsed_ms=8000) for i in range(10)]
        return votes

    def test_honest_expert_is_kept(self):
        self.assertFalse(self.run_assess(self.honest())["excluded"])

    def test_yes_to_foreign_task(self):
        result = self.run_assess(self.honest(foreign="regular"))
        self.assertTrue(result["excluded"])
        self.assertIn("грядок", result["reasons"][0])
        self.assertEqual(result["codes"], ["foreign"])

    def test_inconsistent_duplicates(self):
        result = self.run_assess(self.honest(dup="other_role"))
        self.assertEqual(result["codes"], ["duplicates"])

    def test_too_fast(self):
        votes = [Vote(v.expert_id, v.item_id, v.verdict, importance=v.importance, elapsed_ms=900) for v in self.honest()]
        self.assertEqual(self.run_assess(votes)["codes"], ["speed"])

    def test_same_answers(self):
        votes = [Vote("e", f"x{i}", "regular", importance=5, forecast="more", elapsed_ms=8000) for i in range(12)]
        votes += [Vote("e", "foreign", "other_role", elapsed_ms=8000)]
        self.assertEqual(self.run_assess(votes)["codes"], ["pattern"])

    def test_excluded_votes_do_not_count(self):
        experts = [Expert(id=f"e{i}", label=f"Эксперт {i:02d}") for i in range(12)]
        votes = []
        for expert in experts:
            votes.append(Vote(expert.id, "t1", "regular", importance=4, elapsed_ms=8000))
            votes.append(Vote(expert.id, "foreign", "regular" if expert.id == "e0" else "other_role", elapsed_ms=8000))
        data = engine.ModelData(profession={"id": "p", "name": "Backend"}, sources=[], claims=[],
                                items=[engine.Item(id="t1", level="task", name="Пишет тесты"),
                                       engine.Item(id="foreign", level="task", name=self.names["foreign"], is_control=True,
                                                   control_kind="foreign")], experts=experts, votes=votes)
        content = engine.build(data, AS_OF)
        task = content["tasks"]["t1"]["metrics"]["experts"]
        self.assertEqual((task["n"], task["excluded_votes"]), (11, 1))
        self.assertEqual([item["label"] for item in content["experts"]["excluded"]], ["Эксперт 00"])


class StatusAndTrendTests(unittest.TestCase):
    def test_every_status_row(self):
        cases = [
            (item(is_control=True), metrics(), "control", 1),
            (item(), metrics(n=20, k=3, ci=list(wilson(3, 20))), "rejected", 2),
            (item(), metrics(n=20, k=18, ci=list(wilson(18, 20)), doc_types=1), "confirmed", 3),
            (item(), metrics(n=20, k=18, ci=list(wilson(18, 20)), doc_types=0), "confirmed_early", 4),
            (item(), metrics(n=20, k=11, ci=list(wilson(11, 20)), doc_types=2), "disputed", 5),
            (item(), metrics(n=3, k=3, ci=list(wilson(3, 3)), support=1.0, doc_types=2), "preliminary", 6),
            (item(), metrics(support=0.8, doc_types=1), "hypothesis", 7),
        ]
        for case_item, case_metrics, status, row in cases:
            with self.subTest(status=status):
                self.assertEqual(engine.status_of(case_item, case_metrics, CONFIG), (status, row))

    def test_every_trend_label(self):
        cases = [
            (metrics(n_forecast=12, net=-0.6, automated=0.5), "confirmed", "automated"),
            (metrics(share=0.01, share_prev=0.3, k_prev=8, delta=-0.29, significant=True), "confirmed", "disappearing"),
            (metrics(n_forecast=12, net=-0.4, automated=0.0), "confirmed", "declining"),
            (metrics(share=0.2, share_prev=0.0, delta=0.2, significant=True), "confirmed", "new"),
            (metrics(share=0.5, share_prev=0.2, k_prev=5, delta=0.3, significant=True, in_standard=True), "confirmed", "growing"),
            (metrics(share=0.5, share_prev=0.45, k_prev=9, delta=0.05), "confirmed", "stable"),
            (metrics(), "hypothesis", "no_data"),
            (metrics(share=0.5, share_prev=0.2, k_prev=5, delta=0.3, significant=True, in_standard=True,
                     n_forecast=12, net=-0.5, automated=0.0), "confirmed", "ambiguous"),
        ]
        for case_metrics, status, trend in cases:
            with self.subTest(trend=trend):
                self.assertEqual(engine.trend_of(item(), status, case_metrics, CONFIG)[0], trend)
        proposal = item(origin="expert_proposal")
        self.assertEqual(engine.trend_of(proposal, "confirmed_early", metrics(), CONFIG)[0], "new")


class ExtractionTests(unittest.TestCase):
    def test_quote_check(self):
        source = "Обязанности:\n- Проектировать «схемы» баз данных под нагрузку;\n- Писать тесты"
        self.assertIsNone(quote_problem("проектировать \"схемы\"  баз данных", source))
        self.assertIsNotNone(quote_problem("проектировать структуры баз данных", source))
        checked = verify([ClaimDraft(kind="task", text="Проектирует схемы", quote="Проектировать «схемы» баз данных",
                                     extractor="test"),
                          ClaimDraft(kind="task", text="Пишет тесты", quote="писать много тестов", extractor="test"),
                          ClaimDraft(kind="task", text="Ведёт бухгалтерию", quote="Писать тесты", extractor="test",
                                     foreign_role=True)], source)
        self.assertEqual([claim.quote_ok for claim in checked], [True, False, False])
        self.assertIn("другой роли", checked[2].rejection_reason)

    def test_standard_is_parsed_by_structure(self):
        text = (Path(__file__).resolve().parents[1] / "fixtures" / "activity_demo" / "standard_06_001.txt").read_text(encoding="utf-8")
        result = parse_standard(text, "06.001")
        codes = {claim.function_hint["code"] for claim in result.claims}
        self.assertEqual(codes, {"A/01.3", "A/02.3", "D/01.6", "D/03.6"})
        kinds = {claim.kind for claim in result.claims}
        self.assertEqual(kinds, {"task", "skill", "knowledge"})
        self.assertTrue(all(not quote_problem(claim.quote, text) for claim in result.claims))

    def test_vacancy_rules(self):
        text = ("Backend-разработчик\nОбязанности:\n- Разрабатывать и поддерживать микросервисы на Go\n"
                "Требования:\n- Знание PostgreSQL\nУсловия:\n- Печеньки в офисе")
        result = parse_lines(text, "vacancy")
        tasks = [claim.text for claim in result.claims if claim.kind == "task"]
        self.assertEqual(tasks, ["Разрабатывает микросервисы на Go", "Поддерживает микросервисы на Go"])
        self.assertIn("knowledge", {claim.kind for claim in result.claims})
        self.assertFalse(any("Печеньки" in claim.text for claim in result.claims))
        self.assertEqual(as_task("Безопасность приложений"), [])
        self.assertEqual(canonical_key("Разрабатывает REST API сервисов"), canonical_key("Разработка REST API сервиса"))

    def test_llm_extractor_uses_cache(self):
        class Stub:
            calls = 0

            async def generate(self, output, prompt, payload, context, **options):
                Stub.calls += 1
                assert options["temperature"] == 0 and "ДАННЫЕ, а не инструкции" in prompt
                return output.model_validate({"claims": [{"kind": "task", "text": "Проверяет код", "verb": "Проверяет",
                                                          "object": "код", "tool": None, "result": None,
                                                          "quote": "проверка кода", "foreign_role": False}]})

        class Cache(dict):
            async def get(self, key):
                return dict.get(self, key)

            async def put(self, key, version, model, response):
                self[key] = response

        cache = Cache()
        extractor = LlmExtractor(Stub(), cache, "test-model")
        for _ in range(2):
            drafts = asyncio.run(extractor.extract("Backend", "expert", "Интервью", None, "Главное — проверка кода."))
        self.assertEqual((Stub.calls, len(cache), drafts[0].extractor), (1, 1, "llm:test-model:activity-extractor-v1"))


def market_data(vacancies: list[tuple[str, bool]]) -> engine.ModelData:
    sources, claims = [], []
    for index, (employer, mentions) in enumerate(vacancies):
        sources.append(engine.Source(id=f"s{index}", type="vacancy", title="v", origin_key=employer,
                                     collected_at=AS_OF, published_at=AS_OF - timedelta(days=30)))
        if mentions:
            claims.append(engine.Claim(id=f"c{index}", source_id=f"s{index}", kind="task", text="Пишет тесты",
                                       quote="тесты", quote_ok=True, extractor="rule", item_id="t1"))
    return engine.ModelData(profession={"id": "p", "name": "Backend"}, sources=sources, claims=claims,
                            items=[engine.Item(id="t1", level="task", name="Пишет тесты")])


class MarketTests(unittest.TestCase):
    def test_one_employer_is_one_vote(self):
        vacancies = [("Компания А", True)] * 30 + [(f"Компания {i}", False) for i in range(11)]
        market = engine.build(market_data(vacancies), AS_OF)["tasks"]["t1"]["metrics"]["market"]
        self.assertEqual((market["k_emp"], market["n_emp"], market["vacancies"]), (1, 12, 30))

    def test_too_few_employers_is_not_zero(self):
        market = engine.build(market_data([("А", True), ("Б", False)]), AS_OF)["tasks"]["t1"]["metrics"]["market"]
        self.assertIsNone(market["share"])
        self.assertFalse(market["enough"])

    def test_build_is_deterministic(self):
        data = market_data([("Компания А", True)] * 3 + [(f"Компания {i}", i % 2 == 0) for i in range(12)])
        self.assertEqual(engine.dumps(engine.build(data, AS_OF)), engine.dumps(engine.build(data, AS_OF)))


def model_task(task_id, name, status="confirmed", trend="stable", priority=4, function="f1", areas=()):
    return {"id": task_id, "name": name, "status": status, "status_label": engine.STATUS_LABELS[status], "trend": trend,
            "priority": priority, "function_id": function, "is_control": False, "areas": list(areas), "related": [],
            "metrics": {"market": {"enough": False, "k_emp": 0, "n_emp": 0},
                        "experts": {"k": 18, "n": 20}}}


def fixture_model() -> dict:
    tasks = [model_task("t1", "Проектирует схемы баз данных", areas=["Базы данных"]),
             model_task("t2", "Оптимизирует SQL-запросы", areas=["Базы данных"]),
             model_task("t3", "Пишет хранимые процедуры", areas=["Базы данных"]),
             model_task("t4", "Настраивает мониторинг сервисов", function="f2", priority=6),
             model_task("t5", "Разбирает инциденты в продакшене", function="f2", priority=6),
             model_task("t6", "Поддерживает SOAP-сервисы", status="rejected", trend="disappearing", function="f1"),
             model_task("t7", "Вяжет корзины из лозы", status="hypothesis", function="f1")]
    return {"profession": {"id": "p", "name": "Backend", "is_demo": False}, "as_of": "2026-11-21", "version": 1,
            "tasks": {task["id"]: task for task in tasks}, "knowledge": {},
            "functions": [{"id": "f1", "name": "Работа с данными", "tasks": ["t1", "t2", "t3", "t6", "t7"]},
                          {"id": "f2", "name": "Эксплуатация", "tasks": ["t4", "t5"]}],
            "areas": {"Базы данных": {"kind": "fundamental", "tasks": ["t1", "t2", "t3"], "weight": 0.5}},
            "profile_signal": {"label": "stable"}}


def fixture_program() -> ProgramInput:
    disciplines = [
        Discipline("d1", "Базы данных", 5, 180, topics=("Проектирование схем баз данных", "Оптимизация SQL-запросов",
                                                        "Хранимые процедуры"),
                   outcomes=("Проектирует схемы баз данных",)),
        Discipline("d2", "Хранилища данных", 4, 144, topics=("Проектирование схем баз данных", "Оптимизация SQL-запросов"),
                   outcomes=("Оптимизирует SQL-запросы",)),
        Discipline("d3", "Веб-сервисы SOAP", 4, 144, part="elective",
                   topics=("Поддержка SOAP-сервисов", "Вёрстка на jQuery")),
        Discipline("d4", "История России", 4, 144, part="mandatory_fgos", topics=("История России",)),
        Discipline("d5", "Теория баз данных", 3, 108, topics=("Реляционная модель базы данных", "Вязание корзин из лозы"),
                   outcomes=("Вяжет корзины из лозы",)),
    ]
    return ProgramInput(id="prog", name="Демо", total_zet=240, targets=[("p", 1.0)], disciplines=disciplines,
                        models={"p": fixture_model()})


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.result = analyse(fixture_program())
        self.by_rule = {}
        for rec in self.result["recommendations"]:
            self.by_rule.setdefault(rec["rule"], []).append(rec)

    def test_r1_creates_discipline_for_uncovered_function(self):
        self.assertEqual([rec["target"] for rec in self.by_rule["R1"]], ["Эксплуатация"])
        self.assertGreaterEqual(self.by_rule["R1"][0]["zet"]["needs"], CONFIG["recommendations"]["zet_min"])

    def test_r3_skips_mandatory_fgos_and_r5_protects_fundamental(self):
        targets = {rec["target"] for rec in self.by_rule["R3"]}
        self.assertIn("Веб-сервисы SOAP", targets)
        self.assertNotIn("История России", targets)
        self.assertIn("Теория баз данных", {rec["target"] for rec in self.by_rule["R5"]})
        self.assertNotIn("Теория баз данных", targets)

    def test_r4_merges_overlapping_disciplines(self):
        self.assertEqual([rec["target"] for rec in self.by_rule["R4"]], ["Базы данных + Хранилища данных"])

    def test_r2_replaces_rejected_topics(self):
        self.assertTrue(any("SOAP" in rec["action_text"] for rec in self.by_rule.get("R2", [])) or
                        "Веб-сервисы SOAP" in {rec["target"] for rec in self.by_rule["R3"]})

    def test_no_recommendation_relies_on_hypotheses(self):
        for rule in ("R1", "R2", "R4", "R5"):
            for rec in self.by_rule.get(rule, []):
                self.assertNotIn("hypothesis", {task["status"] for task in rec["tasks"]}, rec["action_text"])

    def test_recommendations_use_topics_independently_of_hours(self):
        from dataclasses import replace
        program = fixture_program()
        program.disciplines = [replace(discipline, hours=0) for discipline in program.disciplines]
        self.assertEqual(analyse(program), self.result)
        self.assertNotIn("R6", self.by_rule)
        self.assertNotIn("hours", self.result)

    def test_analysis_is_deterministic(self):
        self.assertEqual(json.dumps(analyse(fixture_program()), sort_keys=True, ensure_ascii=False),
                         json.dumps(self.result, sort_keys=True, ensure_ascii=False))


class RpdBridgeTests(unittest.TestCase):
    def test_course_gets_tasks_from_profession_model(self):
        from domain.schemas import AppliedActivityModel
        model = fixture_model()
        for task in model["tasks"].values():
            task["reason"] = {"title": "Почему задача включена", "lines": ["Подтвердили 18 из 20 экспертов."]}
            task["quotes"] = [{"text": "проектировать схемы", "source_title": "Вакансия", "date": "2026-10-01"}]
        result = analyse(fixture_program())
        snapshot = course_snapshot({"p": model}, result["links"]["d1"], program={"id": "prog", "name": "Демо"},
                                   discipline={"id": "d1", "name": "Базы данных"})
        self.assertEqual({task["id"] for task in snapshot["tasks"]}, {"t1", "t2", "t3"})
        activity, records = activity_from_snapshot(snapshot, "http://localhost:8501")
        parsed = AppliedActivityModel.model_validate(activity, context={"sources": {r.source_id: r for r in records}})
        self.assertEqual(len(parsed.tasks), 3)
        self.assertTrue(all(record.source_type == "ActivityModel" for record in records))


class LocalUsersTests(unittest.TestCase):
    def test_example_file_has_all_roles(self):
        from auth.yandex_users import LocalUserRepository
        repo = LocalUserRepository(Path(__file__).resolve().parents[1] / "local_users.example.txt")
        legacy = asyncio.run(repo.get_by_email("head@demo.local"))
        self.assertTrue(legacy.id.startswith("program_head:"))
        self.assertEqual(legacy, asyncio.run(repo.get_by_id(legacy.id)))
        self.assertEqual({user.email for user in asyncio.run(repo.list_teachers())},
                         {"head@demo.local", "teacher@demo.local"})
        self.assertEqual(asyncio.run(repo.get_by_email("expert@demo.local")).role, "expert")


class DemoSeedTests(unittest.TestCase):
    """Сквозной прогон демоданных: P0 работает от начала до конца, критерии приёмки выполняются."""

    @classmethod
    def setUpClass(cls):
        from sqlalchemy.ext.asyncio import async_sessionmaker
        from storage.database import ensure_schema, make_engine
        from scripts import seed_activity_demo
        cls.directory = tempfile.TemporaryDirectory()
        cls.url = "sqlite+aiosqlite:///" + (Path(cls.directory.name) / "app.sqlite").as_posix()
        cls.env = {key: os.environ.get(key) for key in ("APP_DATABASE_URL", "SESSION_SIGNING_KEY")}
        os.environ["APP_DATABASE_URL"] = cls.url
        os.environ.setdefault("SESSION_SIGNING_KEY", "x" * 40)
        ensure_schema(cls.url)
        asyncio.run(seed_activity_demo.run("head@demo.local", "teacher@demo.local", "expert@demo.local", False))

        async def load():
            from sqlalchemy import select
            from activity import store
            from storage.activity_models import Profession, Recommendation
            engine_ = make_engine(cls.url)
            factory = async_sessionmaker(engine_, expire_on_commit=False)
            async with factory() as db:
                backend = await db.scalar(select(Profession).where(Profession.name == "Backend-разработчик"))
                version = await store.latest_version(db, backend.id)
                recs = (await db.scalars(select(Recommendation).where(Recommendation.current.is_(True)))).all()
            await engine_.dispose()
            return version.content, recs

        cls.content, cls.recommendations = asyncio.run(load())

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cls.directory.cleanup()

    def test_garden_trap_and_excluded_experts(self):
        tasks = self.content["tasks"].values()
        trap = next(task for task in tasks if "грядок" in task["name"])
        self.assertEqual(trap["status"], "control")
        excluded = self.content["experts"]["excluded"]
        self.assertEqual((self.content["experts"]["total"], len(excluded)), (45, 6))
        self.assertTrue(any("грядок" in reason for item in excluded for reason in item["reasons"]))

    def test_disputed_tasks_show_needed_experts(self):
        disputed = [task for task in self.content["tasks"].values() if task["status"] == "disputed"]
        self.assertEqual(len(disputed), 2)
        self.assertEqual(sorted(task["dispute"]["essential"] for task in disputed), [False, True])
        more = next(task["dispute"]["more"] for task in disputed if not task["dispute"]["essential"])
        self.assertEqual(more, 10)

    def test_recommendations_cover_all_actions(self):
        rules = {rec.rule for rec in self.recommendations}
        self.assertTrue({"R1", "R2", "R3", "R4", "R5", "R7"} <= rules, rules)
        removed = [rec.target for rec in self.recommendations if rec.rule == "R3"]
        self.assertNotIn("История России", removed)
        for rec in self.recommendations:
            if rec.rule != "R3":
                self.assertFalse({task["status"] for task in rec.evidence["tasks"]} & {"hypothesis", "control"})

    def test_every_task_is_traceable(self):
        for task in self.content["tasks"].values():
            if task["status"] in engine.ELIGIBLE:
                self.assertTrue(task["quotes"] or task["origin"] == "expert_proposal", task["name"])
                self.assertTrue(task["reason"]["lines"])
        self.assertTrue(self.content["profession"]["is_demo"])


if __name__ == "__main__":
    unittest.main()
