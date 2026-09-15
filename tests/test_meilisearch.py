from typing import Annotated
from unittest.mock import AsyncMock

import pytest
from mistralai.search.toolkit.document import (
    ChunkPatch,
    ChunkType,
    Document,
    DocumentChunk,
    DocumentPatch,
)
from mistralai.search.toolkit.embedding import CustomEmbeddingModel
from mistralai.search.toolkit.ingestion import File
from mistralai.search.toolkit.ingestion.extractors import PlainTextExtractor
from mistralai.search.toolkit.ingestion.pipelines import Pipeline
from mistralai.search.toolkit.ingestion.text_splitters import CharacterTextSplitter
from mistralai.search.toolkit.retrieval import KeywordRetriever, QueryEngine
from mistralai.search.toolkit.search import (
    GrepMode,
    KeywordSearchQuery,
    NavigableIndex,
    NavigationDirection,
    PatchableIndex,
    VectorSearchQuery,
    VectorStoreIndex,
)
from mistralai.search.toolkit.search.errors import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    IndexingError,
    SearchError,
    SourceNotFoundError,
)

from mistralai.search.toolkit.plugins.meilisearch import (
    MeilisearchApp,
    MeilisearchCollectionSchema,
    MeilisearchField,
    MeilisearchRetriever,
    MeilisearchSearchQuery,
)
from mistralai.search.toolkit.plugins.meilisearch.mapping import record_id

MODEL = CustomEmbeddingModel(name="test", dimensions=3)
VECTOR = [1.0, 0.1, 0.0]


def document(source="notes.md", n=3, embedded=True):
    return Document(
        source_id=source,
        content="full text",
        metadata={"category": "finance", "year": 2026},
        chunks=[
            DocumentChunk(
                source_id=source,
                locator=f"char:{i * 10}-{(i + 1) * 10}",
                start_offset=i * 10,
                end_offset=(i + 1) * 10,
                content=f"chunk {i} about quarterly revenue and markets",
                embedding=VECTOR if embedded else None,
            )
            for i in range(n)
        ],
    )


async def test_index_replace_delete_and_navigation(make_store):
    store, _, _ = await make_store(embedding_model=MODEL)
    assert isinstance(store, VectorStoreIndex)
    assert isinstance(store, NavigableIndex)
    assert isinstance(store, PatchableIndex)
    doc = document(n=5)
    await store.index_document(doc)
    hits = await store.search(VectorSearchQuery(embedding=VECTOR))
    assert len(hits) == 5
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert all(hit.distance is None for hit in hits)
    nxt = await store.navigate(doc.source_id, 0, 10, NavigationDirection.NEXT, top_k=2)
    prev = await store.navigate(doc.source_id, 30, 40, NavigationDirection.PREVIOUS, top_k=2)
    assert [hit.chunk.start_offset for hit in nxt] == [10, 20]
    assert [hit.chunk.start_offset for hit in prev] == [10, 20]
    assert len(await store.read(doc.source_id, 10, 30)) == 2
    assert await store.navigate(doc.source_id, 40, 50, NavigationDirection.NEXT) == []

    await store.index_document(document(n=2))
    await store.index_document(document(n=2))
    assert len(await store.read(doc.source_id, None, None)) == 2
    assert await store.get_chunk(doc.chunks[-1].id) is None
    await store.delete_document(doc.id)
    with pytest.raises(DocumentNotFoundError):
        await store.delete_document(doc.id)
    with pytest.raises(SourceNotFoundError):
        await store.read(doc.source_id, None, None)
    await store.aclose()


async def test_native_keyword_hybrid_filters_facets(make_store):
    store, _, _ = await make_store(
        embedding_model=MODEL,
        settings={"filterableAttributes": ["metadata.document_category", "metadata.document_year"]},
    )
    doc = document()
    await store.index_document(doc)
    other = document(source="other.md", n=1)
    other = other.model_copy(
        update={"metadata": other.metadata.model_copy(update={"category": "other"})}
    )
    await store.index_document(other)
    query = MeilisearchSearchQuery(
        query="quarterly revenue",
        embedding=VECTOR,
        filter='metadata.document_category = "finance"',
        facets=["metadata.document_category", "metadata.document_year"],
        exclude_ids={doc.chunks[0].id},
    )
    response = await store.search_with_facets(query)
    assert {r.chunk.id for r in response.results} == {c.id for c in doc.chunks[1:]}
    assert response.facet_distribution["metadata.document_category"] == {"finance": 2}
    assert response.facet_stats["metadata.document_year"] == {"min": 2026, "max": 2026}
    assert [r.score for r in response.results] == sorted(
        [r.score for r in response.results],
        reverse=True,
    )
    # A typo exercises Meilisearch's native lexical engine, without a query vector.
    hits = await store.search(
        KeywordSearchQuery(query="reveneu", include_content=False, include_metadata=False)
    )
    assert len(hits) == 4
    assert all(r.chunk.content == "" and r.chunk.metadata == {} for r in hits)
    pure = await store.search(MeilisearchSearchQuery(query="reveneu", semantic_ratio=0))
    assert len(pure) == 4
    result = await QueryEngine(KeywordRetriever(store)).search("revenue")
    assert len(result.results) == 4


async def test_keyword_only_and_missing_vectors(make_store):
    store, _, _ = await make_store()
    assert not isinstance(store, VectorStoreIndex)
    await store.index_document(document(embedded=False))
    assert len(await store.search(KeywordSearchQuery(query="revenue"))) == 3
    vector_store, _, _ = await make_store(embedding_model=MODEL)
    await vector_store.index_document(document(embedded=False))
    assert len(await vector_store.search(KeywordSearchQuery(query="revenue"))) == 3


async def test_patch_preserves_vectors_merges_and_removes_metadata(make_store):
    store, raw, _ = await make_store(embedding_model=MODEL)
    doc = document()
    await store.index_document(doc)
    cid = doc.chunks[0].id
    await store.patch_chunk(cid, ChunkPatch(content="updated", metadata={"a": 1, "remove": 2}))
    await store.patch_chunk(cid, ChunkPatch(metadata={"b": 3, "remove": None}))
    await store.patch_document(
        doc.id, DocumentPatch(metadata={"category": None, "state": "published"})
    )
    hit = await store.get_chunk(cid)
    assert hit.chunk.content == "updated"
    assert hit.chunk.metadata == {
        "a": 1,
        "b": 3,
        "document_year": 2026,
        "document_state": "published",
    }
    for result in await store.search(VectorSearchQuery(embedding=VECTOR)):
        assert result.score == pytest.approx(1.0, abs=1e-3)
    stored = await raw.get_document(record_id(cid), retrieve_vectors=True)
    assert stored["_vectors"]["default"]["regenerate"] is False
    await store.patch_chunk(cid, ChunkPatch(embedding=[0, 0, 1]))
    hits = await store.search(VectorSearchQuery(embedding=[0, 0, 1]))
    assert hits[0].chunk.id == cid
    with pytest.raises(ChunkNotFoundError):
        await store.patch_chunk("missing", ChunkPatch(content="x"))
    with pytest.raises(DocumentNotFoundError):
        await store.patch_document("missing", DocumentPatch())


async def test_grep_exact_tokens_and_chunk_types(make_store):
    store, _, _ = await make_store()
    contents = [
        "the quick, brown fox",
        "brown quick fox",
        "quicker brown fox",
        "quick brown summary",
    ]
    doc = document(n=4, embedded=False)
    chunks = [
        c.model_copy(
            update={
                "content": text,
                "chunk_type": ChunkType.SUMMARY if i == 3 else ChunkType.CONTENT,
            }
        )
        for i, (c, text) in enumerate(zip(doc.chunks, contents))
    ]
    await store.index_document(doc.model_copy(update={"chunks": chunks}))
    assert [r.chunk.start_offset for r in await store.grep(doc.source_id, "QUICK brown")] == [0]
    terms = await store.grep(doc.source_id, "quick brown", mode=GrepMode.TERM)
    assert [r.chunk.start_offset for r in terms] == [0, 10]
    assert await store.grep(doc.source_id, "fox dog", mode=GrepMode.TERM) == []
    assert await store.grep(doc.source_id, " ") == []
    assert len(await store.read(doc.source_id, None, None, content_type=ChunkType.SUMMARY)) == 1
    with pytest.raises(SourceNotFoundError):
        await store.grep("missing", " ")


async def test_custom_fields_patch_and_arbitrary_ids(make_store):
    class ArticleChunk(DocumentChunk):
        section: Annotated[str | None, MeilisearchField()] = None

    class Article(Document):
        chunks: list[ArticleChunk]
        title: Annotated[str | None, MeilisearchField(name="headline")] = None
        secret: Annotated[str | None, MeilisearchField(ignore=True)] = None

    class ArticlePatch(DocumentPatch):
        title: str | None = None

    class ArticleChunkPatch(ChunkPatch):
        section: str | None = None

    store, _, app = await make_store(document_type=Article)
    source = 'folder/é "quoted" AND source_id = evil.md'
    chunk = ArticleChunk(
        id="arbitrary/chunk:1",
        source_id=source,
        locator="char:0-5",
        start_offset=0,
        end_offset=5,
        content="hello",
        section="intro",
    )
    doc = Article(source_id=source, content="hello", chunks=[chunk], title="Title", secret="hidden")
    await store.index_document(doc)
    hit = await store.get_chunk(chunk.id)
    assert hit.chunk.id == chunk.id
    assert hit.chunk.headline == "Title" and hit.chunk.section == "intro"
    assert not hasattr(hit.chunk, "document_secret")
    assert len(await store.read(source, None, None)) == 1
    assert await store.search(KeywordSearchQuery(query="hello", exclude_ids={chunk.id})) == []
    await store.patch_document(doc.id, ArticlePatch(title="Updated"))
    await store.patch_chunk(chunk.id, ArticleChunkPatch(section="body"))
    hit = await store.get_chunk(chunk.id)
    assert hit.chunk.headline == "Updated" and hit.chunk.section == "body"
    # Reapplying configuration retains records.
    await app.create_collection(store._client, store._schema.collection_name)
    assert await store.get_chunk(chunk.id) is not None


async def test_invalid_requests_and_failed_write_preserve_existing(make_store, monkeypatch):
    store, _, _ = await make_store(embedding_model=MODEL)
    doc = document()
    await store.index_document(doc)
    bad = doc.model_copy(
        update={"chunks": [doc.chunks[0].model_copy(update={"embedding": [1, 2]})]}
    )
    with pytest.raises(IndexingError, match="dimension"):
        await store.index_document(bad)
    assert len(await store.read(doc.source_id, None, None)) == 3
    with pytest.raises(IndexingError):
        await store.index_document(doc.model_copy(update={"chunks": []}))
    with pytest.raises(SearchError, match="top_k"):
        await store.search(VectorSearchQuery(embedding=VECTOR, top_k=0))
    with pytest.raises(SearchError, match="max_candidates"):
        await store.search(
            VectorSearchQuery(embedding=VECTOR, approximate_options={"max_candidates": 100})
        )
    with pytest.raises(SearchError, match="query embedding"):
        await store.search(MeilisearchSearchQuery(query="revenue"))
    with pytest.raises(SearchError, match="finite"):
        await store.search(VectorSearchQuery(embedding=[float("nan"), 0, 1]))
    with monkeypatch.context() as patcher:

        async def fail(*args, **kwargs):
            raise RuntimeError("write failed")

        patcher.setattr(store._index, "add_documents", fail)
        with pytest.raises(IndexingError):
            await store.index_document(document(n=1))
    assert len(await store.read(doc.source_id, None, None)) == 3


async def test_provider_ingestion_queries_and_supplied_overrides(make_store, provider):
    config, calls = provider
    store, raw, _ = await make_store(embedder_settings=config)
    assert not isinstance(store, VectorStoreIndex)
    pipeline = Pipeline(
        loader=None,
        extractor=PlainTextExtractor(),
        text_splitter=CharacterTextSplitter(chunk_size=100),
        stores=store,
    )
    doc = await pipeline.run_file(
        File(path="memory://alpha.txt", name="alpha.txt", raw=b"alpha text")
    )
    assert "alpha text" in calls
    response = await QueryEngine(MeilisearchRetriever(store)).search("alpha question")
    assert response.results[0].chunk.id == doc.chunks[0].id
    assert "alpha question" in calls
    before = len(calls)
    await store.search(MeilisearchSearchQuery(query="alpha override", embedding=VECTOR))
    assert len(calls) == before
    await store.search(KeywordSearchQuery(query="alpha"))
    assert len(calls) == before
    await store.patch_chunk(doc.chunks[0].id, ChunkPatch(content="changed beta"))
    assert "changed beta" in calls
    stored = await raw.get_document(record_id(doc.chunks[0].id), retrieve_vectors=True)
    assert stored["_vectors"]["default"]["regenerate"] is True
    before = len(calls)
    supplied = document(source="supplied.txt", n=1)
    await store.index_document(supplied)
    await store.patch_chunk(supplied.chunks[0].id, ChunkPatch(content="externally embedded"))
    assert len(calls) == before


async def test_local_pipeline_embedding_and_retriever(make_store):
    store, _, _ = await make_store(embedding_model=MODEL)
    with pytest.raises(ValueError, match="embedder"):
        Pipeline(
            loader=None,
            extractor=PlainTextExtractor(),
            text_splitter=CharacterTextSplitter(),
            stores=store,
        )
    # The marker class must activate the real toolkit embedding processor.
    from mistralai.search.toolkit.embedding import Embedder, EmbeddingResult

    class TestEmbedder(Embedder):
        def __init__(self):
            super().__init__("test")

        async def embed(self, texts, context=None):
            return EmbeddingResult(embeddings=[VECTOR for _ in texts], total_tokens=0)

    embedder = TestEmbedder()
    pipeline = Pipeline(
        loader=None,
        extractor=PlainTextExtractor(),
        text_splitter=CharacterTextSplitter(chunk_size=100),
        stores=store,
        embedder=embedder,
    )
    await pipeline.run_file(File(path="memory://local.txt", name="local.txt", raw=b"local revenue"))
    result = await QueryEngine(MeilisearchRetriever(store, embedder)).search("revenue")
    assert len(result.results) == 1
    assert result.results[0].retriever_id is not None
    embedder.embed_query = AsyncMock(side_effect=AssertionError("keyword search must not embed"))
    assert (
        len(await MeilisearchRetriever(store, embedder, semantic_ratio=0).retrieve("revenue")) == 1
    )


async def test_invalid_settings_are_reported(make_store):
    with pytest.raises(IndexingError, match="Failed to configure"):
        await make_store(settings={"rankingRules": ["invalid-rule"]})


async def test_failed_background_task_is_not_success(make_store):
    store, raw, _ = await make_store(embedding_model=MODEL)
    # Valid HTTP request, rejected asynchronously during vector indexing.
    task = await raw.add_documents(
        [
            {"_id": "bad", "content": "bad vector", "_vectors": {"default": [1, 2]}},
        ],
    )
    with pytest.raises(IndexingError, match="failed"):
        await store._wait(task)


async def test_background_failure_does_not_delete_existing_chunks(make_store, monkeypatch):
    store, _, _ = await make_store()
    doc = document(embedded=False)
    await store.index_document(doc)
    # A valid HTTP request whose document is rejected during background indexing.
    add_documents = store._index.add_documents

    async def invalid_replacement(records):
        records[0]["_id"] = "invalid/id"
        return await add_documents(records)

    monkeypatch.setattr(store._index, "add_documents", invalid_replacement)
    with pytest.raises(IndexingError, match="failed"):
        await store.index_document(document(n=1, embedded=False))
    assert len(await store.read(doc.source_id, None, None)) == 3


async def test_keyword_semantic_and_hybrid_relevance(make_store):
    store, _, _ = await make_store(embedding_model=MODEL)
    doc = document(n=2)
    doc = doc.model_copy(
        update={
            "chunks": [
                doc.chunks[0].model_copy(update={"content": "revenue", "embedding": [0, 0, 1]}),
                doc.chunks[1].model_copy(update={"content": "income", "embedding": VECTOR}),
            ]
        }
    )
    await store.index_document(doc)
    lexical = await store.search(MeilisearchSearchQuery(query="revenue", semantic_ratio=0))
    semantic = await store.search(
        MeilisearchSearchQuery(
            query="revenue",
            embedding=VECTOR,
            semantic_ratio=1,
            top_k=1,
        )
    )
    hybrid = await store.search(
        MeilisearchSearchQuery(
            query="revenue",
            embedding=VECTOR,
            semantic_ratio=0.5,
            top_k=2,
        )
    )
    assert lexical[0].chunk.id == doc.chunks[0].id
    assert semantic[0].chunk.id == doc.chunks[1].id
    assert {r.chunk.id for r in hybrid} == {c.id for c in doc.chunks}


async def test_navigation_grep_and_patches_beyond_search_hit_limit(make_store):
    store, _, _ = await make_store()
    doc = document(n=1005, embedded=False)
    doc = doc.model_copy(
        update={
            "chunks": [
                *doc.chunks[:-1],
                doc.chunks[-1].model_copy(update={"content": "unique needle"}),
            ]
        }
    )
    await store.index_document(doc)
    assert len(await store.read(doc.source_id, None, None, top_k=1010)) == 1005
    assert (await store.grep(doc.source_id, "unique needle"))[0].chunk.id == doc.chunks[-1].id
    await store.patch_document(doc.id, DocumentPatch(metadata={"state": "updated"}))
    last = await store.get_chunk(doc.chunks[-1].id)
    assert last.chunk.metadata["document_state"] == "updated"


def test_schema_rejects_ambiguous_mappings_and_duplicate_names():
    class Bad(Document):
        title: Annotated[str, MeilisearchField(name="content")]

    with pytest.raises(ValueError, match="conflicts"):
        MeilisearchCollectionSchema("docs", document_type=Bad)
    with pytest.raises(ValueError, match="more than once"):
        MeilisearchApp([MeilisearchCollectionSchema("docs"), MeilisearchCollectionSchema("docs")])
