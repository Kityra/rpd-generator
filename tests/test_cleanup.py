"""Этап 7: страж чистоты кода — нет неиспользуемых импортов, удалённые функции не вернулись."""
import ast
import unittest
from pathlib import Path

import literature_parser
from orchestration import call_log, rabota_business, relevance_filter


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("activity", "apps", "auth", "domain", "orchestration", "rabota_it", "scripts", "storage", "tests", "workers")


def unused_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    # Имена из строковых аннотаций («"EvidenceClient"») тоже считаются использованными.
    annotations = " ".join(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                           and isinstance(node.value, str) and len(node.value) < 200)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                if name not in {"*", "annotations"} and name not in used and name not in annotations:
                    result.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
    return result


class CleanupTests(unittest.TestCase):
    def test_no_unused_imports_in_application_modules(self):
        files = [path for package in PACKAGES for path in (ROOT / package).rglob("*.py")
                 if path.name != "__init__.py" and "migrations" not in path.parts]
        files.extend(ROOT.glob("*.py"))
        problems = [item for path in files for item in unused_imports(path)]
        self.assertEqual(problems, [])

    def test_removed_dead_code_stays_removed(self):
        self.assertFalse(hasattr(literature_parser, "find_literature"))
        self.assertFalse(hasattr(call_log, "CallRecorder"))
        self.assertFalse(hasattr(relevance_filter, "stems"))
        self.assertFalse(hasattr(rabota_business, "json"))
        self.assertEqual(rabota_business._course_topic('{"title": "ML"}'), "")
        self.assertEqual(rabota_business._course_topic("ML-инженер\nпрофессия ML"), "ML-инженер")


if __name__ == "__main__":
    unittest.main()
