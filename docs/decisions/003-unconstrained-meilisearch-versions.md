# ADR-003: Record tested Meilisearch versions without enforcing constraints

Status: Accepted

The SDK package name below is superseded by [ADR-004](004-native-async-meilisearch-sdk.md);
the unconstrained version policy remains unchanged.

Date: 2026-09-15

The user requested removal of Meilisearch version constraints for now. This
supersedes the server baseline in ADR-001 and the version gate in ADR-002.

The package declares `meilisearch` without version bounds. The lockfile retains a
concrete SDK version for reproducible development. Neither the adapter nor its
integration tests rejects a server based on its release number. Python, toolkit,
and Pydantic requirements remain unchanged.

Tested releases are verification evidence, not compatibility bounds. Required
server APIs remain native hybrid search, embedders, filtering, and sorted document
fetches. Unsupported APIs surface their normal errors.

The changelog review identified relevant upstream behavior changes:

- [1.16](https://github.com/meilisearch/meilisearch/releases/tag/v1.16.0): sorted document fetches added.
- [1.42.1](https://github.com/meilisearch/meilisearch/releases/tag/v1.42.1): fix for ignoring `regenerate: false` during embedder settings updates with the legacy indexer.
- [1.44](https://github.com/meilisearch/meilisearch/releases/tag/v1.44.0): REST query embedding timeout now follows `searchCutoffMs`.
- [1.46](https://github.com/meilisearch/meilisearch/releases/tag/v1.46.0): fix for batching deletion-by-filter with document additions/updates.
- [1.50](https://github.com/meilisearch/meilisearch/releases/tag/v1.50.0): escaped-filter fix; document fetches default to network-wide retrieval when experimental sharding is configured.
- [1.51](https://github.com/meilisearch/meilisearch/releases/tag/v1.51.0): fix for legacy shorthand filterable-attribute patterns; experimental upgrade flag renamed to `--upgrade-db`.

These findings are recorded for future compatibility work; no minimum version is
enforced on their basis.

Verification: 14 integration tests passed against the official Meilisearch 1.53.2
binary with SDK 0.43.0 and toolkit 0.0.13. Its SHA-256 matched the published release
asset digest. This validates that combination without restricting other versions.
