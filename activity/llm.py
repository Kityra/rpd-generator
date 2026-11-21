"""LLM-экстрактор поверх клиента Yandex AI Studio проекта: журнал model_calls и кэш ответов."""

from orchestration.call_log import CALL_SCOPE, CallScope, RecordingModelClient
from orchestration.clients import YandexModelClient
from run_rpd import yandex_settings
from storage.workflow import SqlWorkflowStore

from activity.extract import LlmExtractor
from activity.store import SqlExtractionCache


def build_extractor(factory) -> LlmExtractor | None:
    """Экстрактор, если в .env настроен Yandex AI Studio; иначе None — работают только правила."""
    try:
        config = yandex_settings()
    except ValueError:
        return None
    CALL_SCOPE.set(CallScope(stage="activity_extract"))
    client = RecordingModelClient(YandexModelClient(config), SqlWorkflowStore(factory).record_model_call)
    return LlmExtractor(client, SqlExtractionCache(factory), config["YANDEX_MODEL_URI"])
