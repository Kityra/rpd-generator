from typing import Any

from langgraph.types import Command


async def resume_graph(graph: Any, request_id: str, interrupt_id: str, payload: dict) -> dict:
    """Вызывается worker после транзакционной проверки и записи команды в БД."""
    config = {"configurable": {"thread_id": request_id}, "recursion_limit": 100}
    snapshot = await graph.aget_state(config)
    pending = [item for task in snapshot.tasks for item in task.interrupts
               if item.value.get("interrupt_id") == interrupt_id]
    if len(pending) == 1:
        return await graph.ainvoke(Command(resume={pending[0].id: payload}), config)
    submitted = payload.get("submission_id")
    consumed = any(item.get("submission_id") == submitted for item in snapshot.values.get("interview_answers", [])) if submitted else False
    if consumed or snapshot.values.get("consumed_review_id") == interrupt_id:
        if snapshot.next and not any(task.interrupts for task in snapshot.tasks):
            return await graph.ainvoke(None, config)
        return snapshot.values
    raise ValueError("Прерывание отсутствует, неоднозначно или ответ уже заменён.")
