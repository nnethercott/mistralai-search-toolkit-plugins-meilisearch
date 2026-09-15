from collections.abc import Iterable
from copy import deepcopy

from meilisearch_python_sdk import AsyncClient
from meilisearch_python_sdk.errors import MeilisearchApiError
from meilisearch_python_sdk.models.settings import MeilisearchSettings
from mistralai.search.toolkit.search.errors import IndexingError
from pydantic import BaseModel, ConfigDict, Field

from ._tasks import wait_for_task
from .index import MeilisearchStoreIndex, MeilisearchVectorStoreIndex
from .schema import MeilisearchCollectionSchema


class MeilisearchConnectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str = "http://localhost:7700"
    api_key: str | None = Field(default=None, repr=False)
    timeout: int = Field(default=60, gt=0)
    task_timeout_ms: int = Field(default=120_000, gt=0)

    def client(self) -> AsyncClient:
        return AsyncClient(self.url, self.api_key, timeout=self.timeout)


class MeilisearchApp:
    def __init__(self, schemas: Iterable[MeilisearchCollectionSchema]) -> None:
        self._schemas = {}
        for schema in schemas:
            if schema.collection_name in self._schemas:
                raise ValueError(
                    f"collection {schema.collection_name!r} is declared more than once"
                )
            self._schemas[schema.collection_name] = schema

    def collection(self, name: str) -> MeilisearchCollectionSchema:
        try:
            return self._schemas[name]
        except KeyError:
            raise ValueError(
                f"unknown collection {name!r}; known: {sorted(self._schemas)}"
            ) from None

    def get_search_index(
        self, src: MeilisearchConnectionConfig | AsyncClient, collection_name: str
    ) -> MeilisearchStoreIndex:
        schema = self.collection(collection_name)
        cls = MeilisearchVectorStoreIndex if schema.embedding_model else MeilisearchStoreIndex
        if isinstance(src, MeilisearchConnectionConfig):
            return cls(src.client(), schema, owns_client=True, task_timeout_ms=src.task_timeout_ms)
        return cls(src, schema)

    async def create_collection(
        self, src: MeilisearchConnectionConfig | AsyncClient, collection_name: str
    ) -> None:
        """Create/configure an index and wait for settings to take effect.

        Existing records are retained. Reapplying a schema is safe; changing an
        embedder's settings may trigger Meilisearch's automatic re-embedding.
        """
        schema = self.collection(collection_name)
        owns = isinstance(src, MeilisearchConnectionConfig)
        client = src.client() if owns else src
        timeout_ms = src.task_timeout_ms if owns else 120_000
        try:
            try:
                index = await client.get_index(collection_name)
            except MeilisearchApiError as exc:
                if exc.code != "index_not_found":
                    raise
                index = await client.create_index(
                    collection_name, primary_key="_id", timeout_in_ms=timeout_ms
                )
            if index.primary_key != "_id":
                raise IndexingError("Existing index must use the plugin's '_id' primary key")

            current = (await index.get_settings()).model_dump(by_alias=True)
            settings = deepcopy(schema.settings)
            settings.setdefault("searchableAttributes", ["content"])
            for name, required in (
                (
                    "filterableAttributes",
                    [
                        "_id",
                        "id",
                        "document_id",
                        "source_id",
                        "locator",
                        "chunk_type",
                        "start_offset",
                        "end_offset",
                    ],
                ),
                ("sortableAttributes", ["start_offset", "end_offset"]),
            ):
                values = list(current.get(name) or [])
                for value in [*settings.get(name, []), *required]:
                    if value not in values:
                        values.append(value)
                settings[name] = values
            if (config := schema.embedder_config()) is not None:
                settings["embedders"] = {schema.embedder_name: config}
            # SDK models otherwise silently discard unknown settings/provider fields.
            task = await index.update_settings(
                MeilisearchSettings.model_validate(settings, extra="forbid")
            )
            await wait_for_task(client, task, timeout_ms)
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(f"Failed to configure collection {collection_name!r}") from exc
        finally:
            if owns:
                await client.aclose()
