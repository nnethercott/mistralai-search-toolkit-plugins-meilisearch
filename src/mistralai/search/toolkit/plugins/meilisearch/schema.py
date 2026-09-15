"""Schema and custom field mapping, following the Qdrant plugin's conventions."""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from mistralai.search.toolkit.document import Document, DocumentChunk
from mistralai.search.toolkit.embedding import (
    DistanceMetric,
    EmbeddingModel,
    MistralEmbeddingPreset,
)


@dataclass(frozen=True)
class MeilisearchField:
    name: str | None = None
    ignore: bool = False


@dataclass(frozen=True)
class CustomFieldMapping:
    key: str
    field: str
    source: Literal["chunk", "document"]


def custom_field_mappings(document_type: type[Document]) -> list[CustomFieldMapping]:
    chunk_type = DocumentChunk
    args = get_args(document_type.model_fields["chunks"].annotation)
    if args:
        element = args[0]
        if hasattr(element, "__metadata__"):
            element = get_args(element)[0]
        if isinstance(element, type) and issubclass(element, DocumentChunk):
            chunk_type = element

    mappings = []
    used = {
        "id",
        "_id",
        "_vectors",
        "document_id",
        "source_id",
        "locator",
        "parent_ref",
        "chunk_type",
        "start_offset",
        "end_offset",
        "content",
        "metadata",
    }
    for model, base, source, prefix in (
        (document_type, Document, "document", "document_"),
        (chunk_type, DocumentChunk, "chunk", ""),
    ):
        for name in sorted(model.model_fields.keys() - base.model_fields.keys()):
            override = next(
                (m for m in model.model_fields[name].metadata if isinstance(m, MeilisearchField)),
                MeilisearchField(),
            )
            if override.ignore:
                continue
            key = override.name or f"{prefix}{name}"
            if key in used or key.startswith("_"):
                raise ValueError(f"Custom field {key!r} conflicts with an indexed field")
            used.add(key)
            mappings.append(CustomFieldMapping(key=key, field=name, source=source))
    return mappings


@dataclass(frozen=True)
class MeilisearchCollectionSchema:
    """One Meilisearch index containing one record per toolkit chunk.

    ``embedding_model`` activates toolkit-side embedding in Pipeline. Alternatively,
    ``embedder_settings`` accepts a native Meilisearch provider configuration.
    With neither, the index supports keyword search only. ``settings`` contains
    native index settings such as filterableAttributes and searchableAttributes.
    """

    collection_name: str
    document_type: type[Document] = Document
    embedding_model: EmbeddingModel | MistralEmbeddingPreset | None = None
    embedder_name: str = "default"
    embedder_settings: dict[str, Any] | None = field(default=None, repr=False)
    settings: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.collection_name):
            raise ValueError("collection_name must contain only ASCII letters, digits, - or _")
        if not self.embedder_name.strip():
            raise ValueError("embedder_name must not be empty")
        model = self.embedding_model
        if isinstance(model, MistralEmbeddingPreset):
            model = model.build_embedding_model()
            object.__setattr__(self, "embedding_model", model)
        if model is not None and model.distance_metric != DistanceMetric.COSINE:
            raise ValueError(
                "Meilisearch uses cosine similarity; other distance metrics are unsupported"
            )
        if "embedders" in self.settings:
            raise ValueError("Use embedder_settings, not settings['embedders']")
        config = self.embedder_config()
        if config is not None:
            if not config.get("source"):
                raise ValueError("embedder_settings requires a source")
            if config["source"] == "userProvided" and not config.get("dimensions"):
                raise ValueError("userProvided embedders require dimensions")
            if model is not None and config.get("dimensions", model.dimensions) != model.dimensions:
                raise ValueError("Embedder dimensions must match embedding_model")
        self.custom_mappings()

    def embedder_config(self) -> dict[str, Any] | None:
        if self.embedder_settings is not None:
            return dict(self.embedder_settings)
        if self.embedding_model is not None:
            return {"source": "userProvided", "dimensions": self.embedding_model.dimensions}
        return None

    def custom_mappings(self) -> list[CustomFieldMapping]:
        return custom_field_mappings(self.document_type)
