# mistralai-search-toolkit-plugins-meilisearch

Meilisearch backend for [Mistral Search Toolkit](https://docs.mistral.ai/en/studio/search/search-toolkit),
closely following [the Qdrant plugin](https://github.com/qdrant-labs/mistral-qdrant-plugin).
Stores one record per chunk and supports native hybrid search, keyword search, filters,
facets, navigation, and patches. Embeddings may come from the toolkit or from a
Meilisearch-configured provider. This is an independent integration.

Uses [`meilisearch-python-sdk`](https://github.com/sanders41/meilisearch-python-sdk)'s
native `AsyncClient`; all network operations are awaited directly.

Requires Python 3.12–3.14. No Meilisearch server-version gate or Python SDK version
bound is imposed. The lockfile records the SDK version used for development;
server and SDK version numbers are independent. The adapter uses native hybrid
search, embedders, filters, and sorted document fetches, so the server must support
those APIs.

## Install

From this directory:

```bash
uv sync
# Or install this local package into another project:
# uv add /path/to/mistralai-search-toolkit-plugins-meilisearch
```

Download a binary for your platform from the
[Meilisearch releases](https://github.com/meilisearch/meilisearch/releases).
Start a local server in another terminal:

```bash
./meilisearch --version
./meilisearch --http-addr 127.0.0.1:7700 --no-analytics
```

If your environment configures a private package index with older toolkit releases,
use `uv sync --index https://pypi.org/simple` to resolve the published dependencies.

## Supplied embeddings: Qdrant-style usage

```python
import asyncio
from contextlib import aclosing

from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.embedding import CustomEmbeddingModel
from mistralai.search.toolkit.plugins.meilisearch import (
    MeilisearchApp,
    MeilisearchCollectionSchema,
    MeilisearchConnectionConfig,
    MeilisearchSearchQuery,
)
from mistralai.search.toolkit.search import KeywordSearchQuery, VectorSearchQuery


async def main():
    schema = MeilisearchCollectionSchema(
        collection_name="docs",
        document_type=Document,
        embedding_model=CustomEmbeddingModel(name="my-embedder", dimensions=3),
        settings={
            "filterableAttributes": ["metadata.document_category"],
        },
    )
    app = MeilisearchApp([schema])
    config = MeilisearchConnectionConfig(url="http://localhost:7700")
    await app.create_collection(config, "docs")
    async with aclosing(app.get_search_index(config, "docs")) as store:
        text = "quarterly revenue grew"
        vector = [0.8, 0.1, 0.2]  # Illustrative; use your embedder's actual output.
        await store.index_document(Document(
            source_id="notes.md", content=text, metadata={"category": "finance"},
            chunks=[DocumentChunk(
                source_id="notes.md", locator=f"char:0-{len(text)}",
                start_offset=0, end_offset=len(text), content=text, embedding=vector,
            )],
        ))

        # Vector only. Adding query="..." to VectorSearchQuery enables native hybrid.
        await store.search(VectorSearchQuery(embedding=vector, top_k=10))
        # Native typo-tolerant keyword search; no query embedding required.
        await store.search(KeywordSearchQuery(query="reveneu", top_k=10))
        # Hybrid with filters and native facet counts.
        response = await store.search_with_facets(MeilisearchSearchQuery(
            query="quarterly revenue", embedding=vector, semantic_ratio=0.5,
            filter='metadata.document_category = "finance"',
            facets=["metadata.document_category"],
        ))
        print(response.results, response.facet_distribution, response.facet_stats)


asyncio.run(main())
```

Like Qdrant, `store.aclose()` closes only clients created from a connection config.
`create_collection(config, ...)` closes its temporary client even if setup fails.
You may instead supply a `meilisearch_python_sdk.AsyncClient` and manage its lifecycle:

```python
from meilisearch_python_sdk import AsyncClient

async with AsyncClient("http://localhost:7700", timeout=60) as client:
    await app.create_collection(client, "docs")
    store = app.get_search_index(client, "docs")
    await store.search(KeywordSearchQuery(query="revenue"))
    # Closing this store leaves the shared client open; the context closes it.
    await store.aclose()
```

The old `meilisearch.Client` is no longer accepted. Connection `timeout` remains in
seconds; `task_timeout_ms` remains in milliseconds.

`embedding_model` configures a `userProvided` embedder by default. Supplied chunk
vectors become `_vectors[embedder_name] = {"embeddings": vector, "regenerate": false}`.
They are preserved during content/metadata patches; update the embedding yourself
when changing externally embedded content. Chunks without vectors remain available
for keyword search. They do not gain semantic embeddings automatically.

## Meilisearch-managed embeddings

Supply native `embedder_settings` to let Meilisearch embed chunks during background
indexing tasks and embed text queries at search time. For example, the documented
Mistral REST embedder configuration is:

```python
import asyncio
import os
from contextlib import aclosing

from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.plugins.meilisearch import (
    MeilisearchApp, MeilisearchCollectionSchema, MeilisearchConnectionConfig,
    MeilisearchRetriever, MeilisearchSearchQuery,
)
from mistralai.search.toolkit.retrieval import QueryEngine

schema = MeilisearchCollectionSchema(
    collection_name="managed_docs",
    embedder_name="mistral",
    embedder_settings={
        "source": "rest",
        "apiKey": os.environ["MISTRAL_API_KEY"],
        "url": "https://api.mistral.ai/v1/embeddings",
        "dimensions": 1024,
        "documentTemplate": "{{doc.content}}",
        "request": {"model": "mistral-embed", "input": ["{{text}}", "{{..}}"]},
        "response": {"data": [{"embedding": "{{embedding}}"}, "{{..}}"]},
    },
)
app = MeilisearchApp([schema])
config = MeilisearchConnectionConfig(task_timeout_ms=300_000)

async def main():
    await app.create_collection(config, "managed_docs")
    async with aclosing(app.get_search_index(config, "managed_docs")) as store:
        text = "quarterly revenue grew"
        await store.index_document(Document(
            source_id="notes.md", content=text,
            chunks=[DocumentChunk(
                source_id="notes.md", locator=f"char:0-{len(text)}",
                start_offset=0, end_offset=len(text), content=text,
            )],
        ))
        response = await QueryEngine(MeilisearchRetriever(store)).search("revenue growth")
        print(response.results)


asyncio.run(main())
```

Native OpenAI, Hugging Face, Ollama, REST, and user-provided embedder configurations
are sent through the SDK; the plugin does not implement providers. Settings dictionaries
use native camelCase keys and are validated with the SDK's `MeilisearchSettings` model.
Unsupported settings/provider fields raise rather than being silently dropped. The REST path
is exercised in integration tests using a local provider. A supplied chunk vector
overrides generation for that chunk; a supplied query vector bypasses query embedding.
Provider-managed chunks use `regenerate: true` so updates can regenerate their vectors.

Every awaited mutation waits for Meilisearch task completion and raises on failure
or cancellation. `task_timeout_ms` controls that wait; a timeout does not cancel the
server task. Reapplying an unchanged schema retains records. Changing embedder settings
may trigger server-side re-embedding, as in native Meilisearch.

## Toolkit pipeline and query engine

Use the regular toolkit `Pipeline` with `stores=store`:

- With `embedding_model`, the app returns a `MeilisearchVectorStoreIndex`. Pass an
  embedder to `Pipeline` and use the toolkit's `VectorRetriever(store, embedder)` or
  `MeilisearchRetriever(store, embedder)` for native hybrid retrieval.
- With provider `embedder_settings` only, the app returns a `MeilisearchStoreIndex`.
  Omit the pipeline embedder; use `MeilisearchRetriever(store)` for hybrid retrieval.
- With neither configuration, ingestion and `KeywordRetriever(store)` work without
  any embedding provider.

This distinction is required because toolkit 0.0.13 uses `isinstance(VectorStoreIndex)`
to enable its local embedding stage. Both classes share the same implementation.
You can supply both an `embedding_model` and provider settings when a local embedding
stage should override the provider using the same model and dimensions.

## Search and facets

`search()` returns the toolkit's `list[SearchResult]`. `search_with_facets()` returns
`MeilisearchSearchResponse`, containing those results plus `facet_distribution`,
`facet_stats`, `estimated_total_hits`, and `processing_time_ms`.

| Query | Behavior |
| --- | --- |
| `KeywordSearchQuery(query=...)` | Always native keyword search |
| `VectorSearchQuery(embedding=...)` | Semantic search |
| `VectorSearchQuery(embedding=..., query=...)` | Native hybrid, ratio 0.5 |
| `MeilisearchSearchQuery(query=...)` | Hybrid with a provider; keyword with no embedder |
| `MeilisearchSearchQuery(..., semantic_ratio=0)` | Keyword only; no embedding call |
| `MeilisearchSearchQuery(..., semantic_ratio=1)` | Semantic only |

For `userProvided` embedders, semantic/hybrid queries require an embedding.
`exclude_ids` filters chunks before both search branches. `matching_strategy` and
`attributes_to_search_on` expose native keyword controls. Configure searchable fields,
synonyms, typo tolerance, ranking rules, filterable fields, and faceting through the
schema's native `settings` dictionary.

Facets count **chunks**, consistent with the Qdrant layout, not unique source files.
Document metadata is copied to each chunk as `metadata.document_<key>`; chunk metadata
uses `metadata.<key>`. Native filter strings and AND/OR arrays are accepted. Faceted
fields must be in `filterableAttributes`. Meilisearch's native hit/facet limits and
hybrid facet semantics apply; counts and estimates are passed through unchanged.
Facet data is available through the plugin method, not the toolkit's `QueryEngine`
result model.

Scores use Meilisearch `_rankingScore` (higher is better). `distance` is always `None`:
a ranking score is not a raw cosine distance. Non-cosine embedding models and explicit
`max_candidates` are rejected because Meilisearch has no equivalent Qdrant controls.

## Navigation, patches, and custom fields

The Qdrant-style methods are available: `navigate`, `read`, `grep`, `get_chunk`,
`patch_chunk`, and `patch_document`. Offsets are `[start, end)`. Navigation and reads
return chunks in source order, without relevance scores. Missing sources/documents
raise the toolkit's corresponding errors; `get_chunk` returns `None` when absent.

`grep` tokenizes Unicode words case-insensitively: `PHRASE` matches adjacent tokens,
`TERM` requires all tokens in any order. It scans the source's records via Meilisearch's
document API so typo tolerance, stop words, and search pagination limits cannot change
these semantics. Cost is proportional to the source size; use native keyword search
for ranked lexical retrieval.

```python
from typing import Annotated
from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.plugins.meilisearch import MeilisearchField

class ArticleChunk(DocumentChunk):
    section: Annotated[str | None, MeilisearchField()] = None

class Article(Document):
    chunks: list[ArticleChunk]
    title: Annotated[str | None, MeilisearchField(name="headline")] = None
    internal_note: Annotated[str | None, MeilisearchField(ignore=True)] = None
```

Pass `document_type=Article`. Chunk fields keep their names; document fields receive
a `document_` prefix unless renamed. Subclass `ChunkPatch`/`DocumentPatch` with matching
fields to patch them. Metadata patches merge keys; explicit `None` removes a key.
Meilisearch's partial update replaces the merged metadata object and preserves vectors.

Canonical chunk IDs stay in `id`. A separate hashed `_id` is the Meilisearch primary
key, allowing arbitrary toolkit IDs. `index_document` writes the new chunks before
deleting stale chunks belonging to that document. These two tasks are not transactional;
serialize concurrent replacements/patches of the same source in the application.
Empty documents are rejected; use `delete_document` explicitly.

## Development

Run a disposable test server with loopback enabled for the fake REST provider:

```bash
meili_test_dir=$(mktemp -d)
./meilisearch --db-path "$meili_test_dir/data.ms" --http-addr 127.0.0.1:17700 \
  --no-analytics --experimental-allowed-ip-networks 127.0.0.1/32
```

In another terminal:

```bash
uv sync
MEILISEARCH_URL=http://127.0.0.1:17700 uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

Tests create uniquely named indexes and delete only those indexes during teardown.
They cover native retrieval, facets, replacements, navigation beyond the search hit
limit, metadata patches, vector preservation, local and provider-driven pipelines,
and asynchronous task failures. A configured server must be running; integration
tests fail rather than silently skip. Tests do not reject servers based on version.
No paid embedding calls are needed.

Verified with Meilisearch **1.53.2**, `meilisearch-python-sdk` **7.5.2**, and toolkit
**0.0.13**. The suite also checks async client ownership, cleanup after setup failures,
task cancellation, and timeouts. These are tested versions,
not required Meilisearch version bounds.

See [the design decision](docs/decisions/001-qdrant-compatible-meilisearch-adapter.md)
and [the version policy](docs/decisions/003-unconstrained-meilisearch-versions.md),
plus [the async SDK migration](docs/decisions/004-native-async-meilisearch-sdk.md).
[NOTICE](NOTICE) records the upstream reference and attribution.
