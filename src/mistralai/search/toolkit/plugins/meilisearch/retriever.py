from mistralai.search.toolkit.context import RetrievalContext
from mistralai.search.toolkit.embedding import Embedder
from mistralai.search.toolkit.retrieval import Retriever
from mistralai.search.toolkit.retrieval.errors import RetrieverException
from mistralai.search.toolkit.search import SearchResult

from .index import MeilisearchSearchQuery, MeilisearchStoreIndex


class MeilisearchRetriever(Retriever):
    """QueryEngine adapter for native hybrid search, optionally embedding locally.

    Without an embedder, Meilisearch generates query vectors using its configured
    provider. Use semantic_ratio=0 for keyword-only retrieval.
    """

    def __init__(
        self,
        client: MeilisearchStoreIndex,
        embedder: Embedder | None = None,
        *,
        semantic_ratio: float = 0.5,
        filter: str | list[str | list[str]] | None = None,
    ) -> None:
        super().__init__()
        MeilisearchSearchQuery(semantic_ratio=semantic_ratio)
        self.client = client
        self.embedder = embedder
        self.semantic_ratio = semantic_ratio
        self.filter = filter

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_metadata: bool = True,
        include_content: bool = True,
        context: RetrievalContext = RetrievalContext(),
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        try:
            embedding = None
            if self.embedder is not None and self.semantic_ratio > 0:
                embedding = await self.embedder.embed_query(query, context=context)
            results = await self.client.search(
                MeilisearchSearchQuery(
                    query=query,
                    embedding=embedding,
                    semantic_ratio=self.semantic_ratio,
                    filter=self.filter,
                    top_k=top_k,
                    include_content=include_content,
                    include_metadata=include_metadata,
                    exclude_ids=exclude_ids or set(),
                ),
                context=context,
            )
            return [r.model_copy(update={"retriever_id": self.retriever_id}) for r in results]
        except Exception as exc:
            raise RetrieverException("Meilisearch retrieval failed") from exc
