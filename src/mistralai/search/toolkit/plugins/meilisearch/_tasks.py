"""Wait for mutations, including canceled tasks the SDK waiter does not handle."""

import asyncio

from meilisearch_python_sdk import AsyncClient
from meilisearch_python_sdk.models.task import TaskInfo
from mistralai.search.toolkit.search.errors import IndexingError


async def wait_for_task(client: AsyncClient, task: TaskInfo, timeout_ms: int) -> None:
    try:
        async with asyncio.timeout(timeout_ms / 1000):
            while True:
                result = await client.get_task(task.task_uid)
                if result.status == "succeeded":
                    return
                if result.status in {"failed", "canceled"}:
                    raise IndexingError(
                        f"Meilisearch task {task.task_uid} {result.status}: {result.error}"
                    )
                await asyncio.sleep(0.05)
    except TimeoutError as exc:
        raise IndexingError(
            f"Meilisearch task {task.task_uid} timed out after {timeout_ms}ms"
        ) from exc
