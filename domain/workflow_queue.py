"""Shared queue limits and safe explanations for the teacher portal."""

MAX_ATTEMPTS = 5

QUEUED_COURSE_STATUS = {
    "create": "queued",
    "relevance": "queued_audit",
    "resume": "answer_queued",
    "review": "review_queued",
}

ERROR_LABELS = {
    "rate_limited": "Внешний сервис ограничил частоту запросов.",
    "upstream_unavailable": "Внешний сервис временно недоступен.",
    "timeout": "Истекло время ожидания внешнего сервиса.",
    "network": "Нет связи с внешним сервисом.",
    "database_unavailable": "База временно недоступна или занята.",
    "authorization": "Проверьте доступ к внешнему сервису.",
    "invalid_configuration": "Проверьте входные данные и настройки.",
    "invalid_response": "Ответ внешнего сервиса не прошёл проверку.",
    "moderation": "Внешний сервис отклонил обработку запроса.",
    "processing_failed": "Обработка завершилась неожиданной ошибкой.",
}
