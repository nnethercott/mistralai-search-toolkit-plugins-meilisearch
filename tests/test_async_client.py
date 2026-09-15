import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from meilisearch_python_sdk import AsyncClient
from meilisearch_python_sdk.models.settings import MeilisearchSettings
from meilisearch_python_sdk.models.task import TaskInfo, TaskResult
from mistralai.search.toolkit.search.errors import IndexingError

from mistralai.search.toolkit.plugins.meilisearch import (
    MeilisearchApp,
    MeilisearchCollectionSchema,
    MeilisearchConnectionConfig,
)
from mistralai.search.toolkit.plugins.meilisearch._tasks import wait_for_task

TASK = TaskInfo(
    task_uid=42, status="enqueued", type="documentAdditionOrUpdate", enqueued_at=datetime.now(UTC)
)


def task_result(status):
    return TaskResult(
        uid=TASK.task_uid,
        status=status,
        type=TASK.task_type,
        enqueued_at=TASK.enqueued_at,
        error={"message": "rejected"} if status == "failed" else None,
    )


async def test_connection_config_uses_async_sdk_and_timeout_seconds():
    config = MeilisearchConnectionConfig(url="http://meili.invalid", timeout=17)
    async with config.client() as client:
        assert isinstance(client, AsyncClient)
        assert client.http_client.timeout.connect == 17
        assert str(client.http_client.base_url) == "http://meili.invalid"
    assert client.http_client.is_closed


@pytest.mark.parametrize("owns", [True, False])
async def test_store_closes_only_owned_clients(owns, monkeypatch):
    config = MeilisearchConnectionConfig(task_timeout_ms=1234)
    async with config.client() as client:
        monkeypatch.setattr(MeilisearchConnectionConfig, "client", lambda self: client)
        app = MeilisearchApp([MeilisearchCollectionSchema("docs")])
        store = app.get_search_index(config if owns else client, "docs")
        assert store._client is client
        assert store._task_timeout_ms == (1234 if owns else 120_000)
        await store.aclose()
        await store.aclose()
        assert client.http_client.is_closed is owns


@pytest.mark.parametrize("owns", [True, False])
@pytest.mark.parametrize("outcome", ["success", "failure", "canceled"])
async def test_provisioning_client_cleanup(owns, outcome, monkeypatch):
    config = MeilisearchConnectionConfig()
    async with config.client() as client:
        monkeypatch.setattr(MeilisearchConnectionConfig, "client", lambda self: client)
        index = client.index("docs")
        index.primary_key = "_id"
        monkeypatch.setattr(client, "get_index", AsyncMock(return_value=index))
        monkeypatch.setattr(index, "get_settings", AsyncMock(return_value=MeilisearchSettings()))
        update = AsyncMock(return_value=TASK)
        if outcome == "failure":
            update.side_effect = RuntimeError("settings unavailable")
        elif outcome == "canceled":
            update.side_effect = asyncio.CancelledError()
        monkeypatch.setattr(index, "update_settings", update)
        monkeypatch.setattr(client, "get_task", AsyncMock(return_value=task_result("succeeded")))
        app = MeilisearchApp([MeilisearchCollectionSchema("docs")])
        src = config if owns else client
        if outcome == "success":
            await app.create_collection(src, "docs")
            settings = update.call_args.args[0]
            assert isinstance(settings, MeilisearchSettings)
            assert settings.searchable_attributes == ["content"]
            assert "source_id" in settings.filterable_attributes
        else:
            error = IndexingError if outcome == "failure" else asyncio.CancelledError
            with pytest.raises(error):
                await app.create_collection(src, "docs")
        assert client.http_client.is_closed is owns


@pytest.mark.parametrize("status", ["failed", "canceled"])
async def test_terminal_task_errors_fail_promptly(status):
    client = AsyncMock(spec=AsyncClient)
    client.get_task.return_value = task_result(status)
    with pytest.raises(IndexingError, match=f"42 {status}"):
        await wait_for_task(client, TASK, timeout_ms=1000)
    client.get_task.assert_awaited_once_with(42)


async def test_task_waits_for_completion():
    client = AsyncMock(spec=AsyncClient)
    client.get_task.side_effect = [task_result("processing"), task_result("succeeded")]
    await wait_for_task(client, TASK, timeout_ms=1000)
    assert client.get_task.await_count == 2


async def test_task_timeout_bounds_requests_without_canceling_server_task():
    client = AsyncMock(spec=AsyncClient)

    async def stalled_request(*args):
        await asyncio.Event().wait()

    client.get_task.side_effect = stalled_request
    with pytest.raises(IndexingError, match="42 timed out"):
        await wait_for_task(client, TASK, timeout_ms=10)
    client.cancel_tasks.assert_not_called()


async def test_caller_cancellation_propagates():
    client = AsyncMock(spec=AsyncClient)
    client.get_task.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await wait_for_task(client, TASK, timeout_ms=1000)
