import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from meilisearch_python_sdk import AsyncClient

from mistralai.search.toolkit.plugins.meilisearch import (
    MeilisearchApp,
    MeilisearchCollectionSchema,
)


@pytest.fixture
async def meilisearch_client():
    url = os.environ.get("MEILISEARCH_URL", "http://127.0.0.1:7700")
    async with AsyncClient(url, os.environ.get("MEILISEARCH_API_KEY"), timeout=10) as client:
        # Integration tests should fail visibly if the configured server is unavailable.
        await client.health()
        yield client


@pytest.fixture
async def make_store(meilisearch_client):
    client = meilisearch_client
    names = []

    async def create(**kwargs):
        name = f"mst_test_{uuid.uuid4().hex}"
        names.append(name)
        schema = MeilisearchCollectionSchema(collection_name=name, **kwargs)
        app = MeilisearchApp([schema])
        await app.create_collection(client, name)
        return app.get_search_index(client, name), client.index(name), app

    yield create
    for name in names:
        task = await client.index(name).delete()
        await client.wait_for_task(task.task_uid, timeout_in_ms=10_000, raise_for_status=True)


@pytest.fixture
def provider():
    """Real Meilisearch calls this local REST embedder; no external model calls."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            inputs = body["input"]
            calls.extend(inputs)
            vectors = [
                [1.0, 0.1, 0.0] if "alpha" in text.lower() else [0.0, 0.1, 1.0] for text in inputs
            ]
            response = json.dumps({"data": [{"embedding": vec} for vec in vectors]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield (
        {
            "source": "rest",
            "url": f"http://127.0.0.1:{server.server_port}/embeddings",
            "dimensions": 3,
            "documentTemplate": "{{doc.content}}",
            "request": {"input": ["{{text}}", "{{..}}"]},
            "response": {"data": [{"embedding": "{{embedding}}"}, "{{..}}"]},
        },
        calls,
    )
    server.shutdown()
    server.server_close()
    thread.join()
