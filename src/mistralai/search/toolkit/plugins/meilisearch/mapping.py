"""Map toolkit chunks to Meilisearch records, retaining Qdrant's public field layout."""

import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.search import SearchResult, SearchResultChunk

from .schema import CustomFieldMapping


def record_id(chunk_id: str) -> str:
    # A separate, always-hashed primary key avoids collisions with literal UUID ids
    # and supports toolkit ids containing characters Meilisearch does not accept.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def to_record(
    document: Document, chunk: DocumentChunk, custom: Iterable[CustomFieldMapping]
) -> dict[str, Any]:
    metadata = chunk.metadata.model_dump(mode="json", exclude_none=True)
    metadata.update(
        {
            f"document_{k}": v
            for k, v in document.metadata.model_dump(mode="json", exclude_none=True).items()
        }
    )
    record = {
        "_id": record_id(chunk.id),
        "id": chunk.id,
        "document_id": document.id,
        "source_id": chunk.source_id,
        "locator": chunk.locator,
        "parent_ref": chunk.parent_ref,
        "chunk_type": chunk.chunk_type.value,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "content": chunk.content,
        "metadata": metadata,
    }
    document_values = document.model_dump(mode="json", exclude={"chunks"})
    chunk_values = chunk.model_dump(mode="json", exclude={"embedding"})
    for field in custom:
        values = document_values if field.source == "document" else chunk_values
        record[field.key] = values.get(field.field)
    return record


def result(
    record: Mapping[str, Any],
    custom_keys: Iterable[str],
    *,
    score: float = 0.0,
    include_content: bool = True,
    include_metadata: bool = True,
) -> SearchResult:
    return SearchResult(
        chunk=SearchResultChunk(
            id=record["id"],
            source_id=record["source_id"],
            locator=record["locator"],
            parent_ref=record.get("parent_ref"),
            start_offset=record.get("start_offset"),
            end_offset=record.get("end_offset"),
            chunk_type=record["chunk_type"],
            content=record.get("content", "") if include_content else "",
            metadata=(record.get("metadata") or {}) if include_metadata else {},
            **{k: record[k] for k in custom_keys if k in record},
        ),
        score=score,
        # Meilisearch's ranking score is not a raw vector distance.
        distance=None,
    )
