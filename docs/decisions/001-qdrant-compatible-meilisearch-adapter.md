# ADR-001: Follow Qdrant's chunk layout using Meilisearch primitives

Status: Accepted

The server-version baseline below is superseded by
[ADR-002](002-stable-meilisearch-1.53.md); the adapter design remains accepted.
The synchronous SDK/transport choice is superseded by
[ADR-004](004-native-async-meilisearch-sdk.md).

Date: 2026-09-15

## Context

The requested integration follows `mistralai-search-toolkit-plugins-qdrant` 0.1.5
(commit `06f12c7f0d4f74e6e29fc3d0226e4f08bb9e00ae`) closely, with minimal complexity,
native Meilisearch hybrid/keyword search and facets, and both supplied and
provider-generated embeddings. The reference repository and published source match.

## Decision

Retain the app/schema/index/mapping separation, one record per chunk, canonical
identities, metadata prefixes, custom field annotations, replacement cleanup,
navigation, and patch semantics. Use the official synchronous Meilisearch SDK
through `asyncio.to_thread`; serialize calls per store because SDK HTTP headers are
mutable. No alternate HTTP transport, job queue, or provider abstraction is added.

Use native Meilisearch hybrid search (`q`, `vector`, `hybrid.semanticRatio`) in one
request. The reference Qdrant implementation fetches vectors and full-text matches
separately and fuses them with Python RRF. Meilisearch already provides lexical
ranking, hybrid blending, filters, and facets; reproducing RRF would duplicate that.

Supplied vectors set `regenerate: false`. Missing vectors use `regenerate: true`
when a provider is configured, or an empty vector list for `userProvided` so chunks
remain keyword-searchable. The latter deliberately differs from Qdrant's skipping
of unembedded chunks. Native provider settings are passed through without wrappers.

The base store subclasses `KeywordStoreIndex`; a thin `VectorStoreIndex` subclass
marks local-embedding mode. The app chooses it when `embedding_model` is supplied.
This is necessary because toolkit 0.0.13's Pipeline requires and runs a local embedder
for any `VectorStoreIndex`. Provider-managed mode uses the base store and a small
`MeilisearchRetriever` that can submit plain text through QueryEngine.

Preserve `search() -> list[SearchResult]`. An additive `search_with_facets()` returns
the same hits and native facet distributions/statistics. Facets count indexed chunks.
Scores use `_rankingScore`; distance stays unset because scores cannot safely be
treated as raw cosine distances. Unsupported distance metrics and ANN candidate
controls fail explicitly.

Mutations wait for terminal Meilisearch tasks and inspect status; the SDK's wait
method does not raise for failed tasks. Replacement adds records before deleting
stale ones. Patches use native partial updates, merging metadata locally before
replacing that top-level field. No read-back/rewrite of stored vectors is needed.

Navigation uses filtered, sorted, paginated document fetches, independent of search
hit limits. Grep uses those fetches with local token matching to retain exact phrase/
term and reading-order semantics regardless of native typo tolerance and ranking.

## Consequences

The adapter requires Meilisearch 1.43+ for sorted document fetches. It keeps Qdrant's
non-transactional replacement and read/modify/write patch boundaries: applications
should serialize mutations of a source. Grep scans one source; it is not intended
as a replacement for native ranked keyword search. Changes to provider settings may
cause native Meilisearch re-embedding. No fork of the toolkit is required.

## Sources

- [Qdrant reference](https://github.com/qdrant-labs/mistral-qdrant-plugin)
- Docstral: toolkit 0.0.13 backend, index lifecycle, and retrieval contracts
- Context7 and [Meilisearch's embedding docs](https://www.meilisearch.com/docs/capabilities/hybrid_search/how_to/search_with_user_provided_embeddings)
- [Meilisearch Mistral REST provider](https://www.meilisearch.com/docs/capabilities/hybrid_search/providers/mistral)
- Official `meilisearch-python` SDK source and real-server integration tests
