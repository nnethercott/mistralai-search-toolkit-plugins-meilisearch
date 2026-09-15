import json
import math
import re
from typing import Any, Literal

from meilisearch_python_sdk import AsyncClient
from meilisearch_python_sdk.errors import MeilisearchApiError
from meilisearch_python_sdk.models.search import Hybrid
from meilisearch_python_sdk.models.task import TaskInfo
from mistralai.search.toolkit.context import IngestContext, RetrievalContext
from mistralai.search.toolkit.document import ChunkPatch, ChunkType, Document, DocumentPatch
from mistralai.search.toolkit.search import (
    BaseSearchQuery,
    GrepMode,
    KeywordSearchQuery,
    KeywordStoreIndex,
    NavigationDirection,
    SearchResult,
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
from pydantic import BaseModel, Field

from . import mapping
from ._tasks import wait_for_task
from .schema import MeilisearchCollectionSchema


class MeilisearchSearchQuery(BaseSearchQuery):
    """Native search: text, vector, or both; provider embedders can vectorize text.

    semantic_ratio defaults to 0.5 with text and an embedder, 1 for vector-only,
    and 0 without an embedder. Explicit 0 always performs keyword-only search.
    """

    query: str | None = None
    embedding: list[float] | None = None
    semantic_ratio: float | None = Field(default=None, ge=0, le=1)
    filter: str | list[str | list[str]] | None = None
    facets: list[str] = Field(default_factory=list)
    matching_strategy: Literal["last", "all", "frequency"] = "last"
    attributes_to_search_on: list[str] | None = None


class MeilisearchSearchResponse(BaseModel):
    results: list[SearchResult]
    facet_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)
    facet_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    estimated_total_hits: int | None = None
    processing_time_ms: int = 0


Query = MeilisearchSearchQuery | KeywordSearchQuery | VectorSearchQuery


def _eq(key: str, value: str) -> str:
    return f"{key} = {json.dumps(value, ensure_ascii=False)}"


class MeilisearchStoreIndex(KeywordStoreIndex):
    """Chunk store supporting keyword, native hybrid, navigation, and patching.

    The app selects the VectorStoreIndex subclass when embedding_model is supplied,
    which tells the toolkit Pipeline to generate embeddings locally.
    """

    def __init__(
        self,
        client: AsyncClient,
        schema: MeilisearchCollectionSchema,
        *,
        owns_client: bool = False,
        task_timeout_ms: int = 120_000,
    ) -> None:
        self._client = client
        self._owns_client = owns_client
        self._task_timeout_ms = task_timeout_ms
        self._index = client.index(schema.collection_name)
        self._schema = schema
        self._embedder = schema.embedder_config()
        self._custom = schema.custom_mappings()
        self._keys = tuple(m.key for m in self._custom)

    async def _wait(self, task: TaskInfo) -> None:
        await wait_for_task(self._client, task, self._task_timeout_ms)

    def _validate_vector(self, vector: list[float]) -> None:
        if self._embedder is None:
            raise ValueError("Configure an embedder before supplying vectors")
        if not vector or not all(math.isfinite(x) for x in vector):
            raise ValueError("Vectors must be nonempty and contain finite numbers")
        dimensions = self._embedder.get("dimensions")
        if dimensions is not None and len(vector) != dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {dimensions}, got {len(vector)}"
            )

    async def _records(
        self,
        filters: list[str],
        *,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        records = []
        while limit is None or len(records) < limit:
            size = 1000 if limit is None else min(1000, limit - len(records))
            page = await self._index.get_documents(
                filter=filters,
                offset=len(records),
                limit=size,
                sort=f"start_offset:{'desc' if reverse else 'asc'},end_offset:asc",
            )
            records.extend(page.results)
            if len(page.results) < size or len(records) >= page.total:
                break
        return records

    async def _get(self, chunk_id: str) -> dict[str, Any] | None:
        try:
            return await self._index.get_document(mapping.record_id(chunk_id))
        except MeilisearchApiError as exc:
            if exc.code == "document_not_found":
                return None
            raise

    async def _require_source(self, source_id: str) -> None:
        if not await self._records([_eq("source_id", source_id)], limit=1):
            raise SourceNotFoundError(source_id)

    async def index_document(
        self,
        document: Document,
        context: IngestContext = IngestContext(),
    ) -> None:
        if not document.chunks:
            raise IndexingError("No chunks for document; use delete_document() to remove it")
        try:
            records = []
            for chunk in document.chunks:
                record = mapping.to_record(document, chunk, self._custom)
                if chunk.embedding is not None:
                    self._validate_vector(chunk.embedding)
                    record["_vectors"] = {
                        self._schema.embedder_name: {
                            "embeddings": chunk.embedding,
                            "regenerate": False,
                        }
                    }
                elif self._embedder is not None:
                    automatic = self._embedder["source"] != "userProvided"
                    record["_vectors"] = {
                        self._schema.embedder_name: (
                            {"regenerate": True}
                            if automatic
                            else {"embeddings": [], "regenerate": False}
                        )
                    }
                records.append(record)
            await self._wait(await self._index.add_documents(records))
            # A failed write must never delete the previously indexed document.
            keep = json.dumps([r["_id"] for r in records])
            await self._wait(
                await self._index.delete_documents_by_filter(
                    f"{_eq('document_id', document.id)} AND _id NOT IN {keep}"
                )
            )
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(f"Failed to index document: {exc}") from exc

    async def delete_document(
        self,
        doc_id: str,
        context: IngestContext = IngestContext(),
    ) -> None:
        try:
            filters = [_eq("document_id", doc_id)]
            if not await self._records(filters, limit=1):
                raise DocumentNotFoundError(doc_id)
            await self._wait(await self._index.delete_documents_by_filter(filters[0]))
        except (DocumentNotFoundError, IndexingError):
            raise
        except Exception as exc:
            raise IndexingError(f"Failed to delete document {doc_id!r}") from exc

    def _query(self, query: Query) -> tuple[str, dict[str, Any]]:
        if query.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if isinstance(query, VectorSearchQuery):
            if query.approximate_options.max_candidates is not None:
                raise ValueError("Meilisearch does not expose max_candidates")
            q = MeilisearchSearchQuery(**query.model_dump(exclude={"approximate_options"}))
        elif isinstance(query, KeywordSearchQuery):
            q = MeilisearchSearchQuery(**query.model_dump(), semantic_ratio=0)
        else:
            q = query
        text = q.query or ""
        ratio = q.semantic_ratio
        if ratio is None:
            ratio = (0.5 if text.strip() else 1.0) if self._embedder else 0.0
        params: dict[str, Any] = {
            "limit": q.top_k,
            "show_ranking_score": True,
            "matching_strategy": q.matching_strategy,
        }
        if ratio > 0:
            if self._embedder is None:
                raise ValueError("Semantic search requires an embedder")
            if q.embedding is None and self._embedder["source"] == "userProvided":
                raise ValueError("userProvided semantic search requires a query embedding")
            if q.embedding is None and not text.strip():
                raise ValueError("Semantic search requires query text or a vector")
            params["hybrid"] = Hybrid(embedder=self._schema.embedder_name, semantic_ratio=ratio)
            if q.embedding is not None:
                self._validate_vector(q.embedding)
                params["vector"] = q.embedding
        elif q.embedding is not None and self._embedder is None:
            raise ValueError("Configure an embedder before supplying vectors")
        filters = []
        if q.filter:
            filters.extend([q.filter] if isinstance(q.filter, str) else q.filter)
        if q.exclude_ids:
            # Native filters apply to both keyword and semantic candidates.
            ids = [mapping.record_id(cid) for cid in sorted(q.exclude_ids)]
            filters.append(f"_id NOT IN {json.dumps(ids)}")
        if filters:
            params["filter"] = filters
        if q.facets:
            params["facets"] = q.facets
        if q.attributes_to_search_on is not None:
            params["attributes_to_search_on"] = q.attributes_to_search_on
        return text, params

    async def search(
        self,
        query: Query,
        context: RetrievalContext = RetrievalContext(),
    ) -> list[SearchResult]:
        return (await self.search_with_facets(query, context)).results

    async def search_with_facets(
        self,
        query: Query,
        context: RetrievalContext = RetrievalContext(),
    ) -> MeilisearchSearchResponse:
        """Search with native facet distributions and numeric min/max statistics.

        Counts refer to indexed chunks, and retain Meilisearch's native semantics.
        Facet attributes must be configured as filterableAttributes in the schema.
        """
        try:
            text, params = self._query(query)
            response = await self._index.search(text, **params)
            return MeilisearchSearchResponse(
                results=[
                    mapping.result(
                        hit,
                        self._keys,
                        score=hit["_rankingScore"],
                        include_content=query.include_content,
                        include_metadata=query.include_metadata,
                    )
                    for hit in response.hits
                ],
                facet_distribution=response.facet_distribution or {},
                facet_stats=response.facet_stats or {},
                estimated_total_hits=response.estimated_total_hits,
                processing_time_ms=response.processing_time_ms,
            )
        except Exception as exc:
            raise SearchError(f"Search query failed: {exc}", query=query.query) from exc

    async def _span(
        self,
        source_id: str,
        filters: list[str],
        *,
        top_k: int,
        reverse: bool = False,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise SearchError("top_k must be at least 1")
        try:
            records = await self._records(filters, limit=top_k, reverse=reverse)
            if not records:
                await self._require_source(source_id)
            if reverse:
                records.reverse()
            return [mapping.result(record, self._keys) for record in records]
        except SourceNotFoundError:
            raise
        except Exception as exc:
            raise SearchError("Positional query failed") from exc

    async def navigate(
        self,
        source_id: str,
        start_offset: int,
        end_offset: int,
        direction: NavigationDirection,
        *,
        top_k: int = 1,
        content_type: ChunkType = ChunkType.CONTENT,
        context: RetrievalContext = RetrievalContext(),
    ) -> list[SearchResult]:
        direction = NavigationDirection(direction)
        nxt = direction == NavigationDirection.NEXT
        bound = (
            f"start_offset >= {int(end_offset)}" if nxt else f"end_offset <= {int(start_offset)}"
        )
        return await self._span(
            source_id,
            [_eq("source_id", source_id), _eq("chunk_type", content_type.value), bound],
            top_k=top_k,
            reverse=not nxt,
        )

    async def read(
        self,
        source_id: str,
        start_offset: int | None,
        end_offset: int | None,
        *,
        content_type: ChunkType = ChunkType.CONTENT,
        top_k: int = 20,
        context: RetrievalContext = RetrievalContext(),
    ) -> list[SearchResult]:
        filters = [_eq("source_id", source_id), _eq("chunk_type", content_type.value)]
        if start_offset is not None:
            filters.append(f"start_offset >= {int(start_offset)}")
        if end_offset is not None:
            filters.append(f"end_offset <= {int(end_offset)}")
        return await self._span(source_id, filters, top_k=top_k)

    async def get_chunk(
        self,
        chunk_id: str,
        *,
        context: RetrievalContext = RetrievalContext(),
    ) -> SearchResult | None:
        try:
            record = await self._get(chunk_id)
            return mapping.result(record, self._keys) if record is not None else None
        except Exception as exc:
            raise SearchError(f"Failed to fetch chunk {chunk_id!r}") from exc

    async def grep(
        self,
        source_id: str,
        pattern: str,
        *,
        mode: GrepMode = GrepMode.PHRASE,
        content_type: ChunkType = ChunkType.CONTENT,
        top_k: int = 5,
        context: RetrievalContext = RetrievalContext(),
    ) -> list[SearchResult]:
        """Match Unicode word tokens, case-insensitively, in source reading order.

        Scan source records through the document API so typo tolerance, ranking,
        stop words and Meilisearch's search hit limit cannot change grep semantics.
        PHRASE requires adjacent tokens; TERM requires all tokens in any order.
        """
        if top_k < 1:
            raise SearchError("top_k must be at least 1")
        mode = GrepMode(mode)
        try:
            await self._require_source(source_id)
            needles = re.findall(r"\w+", pattern.casefold())
            if not needles:
                return []
            records = await self._records(
                [
                    _eq("source_id", source_id),
                    _eq("chunk_type", content_type.value),
                ]
            )
            results = []
            for record in records:
                words = re.findall(r"\w+", record["content"].casefold())
                matched = (
                    any(words[i : i + len(needles)] == needles for i in range(len(words)))
                    if mode == GrepMode.PHRASE
                    else set(needles).issubset(words)
                )
                if matched:
                    results.append(mapping.result(record, self._keys))
                    if len(results) == top_k:
                        break
            return results
        except SourceNotFoundError:
            raise
        except Exception as exc:
            raise SearchError("Grep query failed") from exc

    def _patch_values(
        self,
        record: dict[str, Any],
        patch: ChunkPatch | DocumentPatch,
        source: str,
    ) -> dict[str, Any]:
        update = {"_id": record["_id"]}
        if patch.metadata is not None:
            metadata = dict(record.get("metadata") or {})
            prefix = "document_" if source == "document" else ""
            for key, value in patch.metadata.model_dump(mode="json", exclude_unset=True).items():
                if value is None:
                    metadata.pop(f"{prefix}{key}", None)
                else:
                    metadata[f"{prefix}{key}"] = value
            update["metadata"] = metadata
        base = ChunkPatch if source == "chunk" else DocumentPatch
        extra = (
            type(patch).model_fields.keys() - base.model_fields.keys()
        ) & patch.model_fields_set
        known = {m.field: m.key for m in self._custom if m.source == source}
        values = patch.model_dump(mode="json", exclude_unset=True)
        for name in extra:
            if name not in known:
                raise IndexingError(f"Patch field {name!r} is not a custom {source} field")
            update[known[name]] = values[name]
        return update

    async def patch_chunk(
        self,
        chunk_id: str,
        patch: ChunkPatch,
        context: IngestContext = IngestContext(),
    ) -> None:
        try:
            record = await self._get(chunk_id)
            if record is None:
                raise ChunkNotFoundError(chunk_id)
            update = self._patch_values(record, patch, "chunk")
            if patch.content is not None:
                update["content"] = patch.content
            if patch.embedding is not None:
                self._validate_vector(patch.embedding)
                update["_vectors"] = {
                    self._schema.embedder_name: {
                        "embeddings": patch.embedding,
                        "regenerate": False,
                    }
                }
            # Partial updates preserve supplied vectors and allow automatic ones
            # to regenerate when their document template's input changes.
            await self._wait(await self._index.update_documents([update]))
        except (ChunkNotFoundError, IndexingError):
            raise
        except Exception as exc:
            raise IndexingError(f"Failed to patch chunk: {exc}") from exc

    async def patch_document(
        self,
        document_id: str,
        patch: DocumentPatch,
        context: IngestContext = IngestContext(),
    ) -> None:
        try:
            records = await self._records([_eq("document_id", document_id)])
            if not records:
                raise DocumentNotFoundError(document_id)
            updates = [self._patch_values(record, patch, "document") for record in records]
            await self._wait(await self._index.update_documents(updates))
        except (DocumentNotFoundError, IndexingError):
            raise
        except Exception as exc:
            raise IndexingError("Failed to patch document") from exc

    async def aclose(self) -> None:
        """Close an owned client; a supplied client's lifecycle belongs to its caller."""
        if self._owns_client:
            await self._client.aclose()


class MeilisearchVectorStoreIndex(MeilisearchStoreIndex, VectorStoreIndex):
    """Marker enabling the toolkit Pipeline's local embedding stage."""
