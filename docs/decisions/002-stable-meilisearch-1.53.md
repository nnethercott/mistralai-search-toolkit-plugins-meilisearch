# ADR-002: Target stable Meilisearch 1.53.x

Status: Superseded by [ADR-003](003-unconstrained-meilisearch-versions.md).

Date: 2026-09-15

## Context

The initial adapter was exercised against the locally installed Meilisearch 1.43.1.
The user requested stable 1.53.x instead. GitHub identifies 1.53.2 as the latest
stable release, published on 2026-09-07.

## Decision

Target stable Meilisearch 1.53.x and use 1.53.2 as the reproducible server baseline.
This supersedes only the server-version baseline in ADR-001. Keep the official
Python SDK at 0.43.0; its version is independent of the server release number.

The integration session checks the server version before creating any indexes.
Versions outside stable 1.53.x, including prereleases, fail explicitly. The README
links to the pinned binary release instead of relying on a system-installed binary.

## Consequences

Validation covers the requested stable release line, including native hybrid search,
facets, automatic REST embeddings, supplied vectors, mutations, and navigation.
Supporting a different server release line requires an intentional test baseline
update. No server-version check is added to normal library operations.

## Sources

- [Meilisearch 1.53.2 release](https://github.com/meilisearch/meilisearch/releases/tag/v1.53.2)
- Context7: current Meilisearch hybrid search and vector-generation contracts
