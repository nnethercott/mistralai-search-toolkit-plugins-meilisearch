# ADR-004: Use the native async Meilisearch Python SDK

Status: Accepted

Date: 2026-09-15

## Context

The user requested `sanders41/meilisearch-python-sdk` in place of the official
synchronous client to simplify the adapter and use native async operations.

## Decision

Replace the `meilisearch` dependency with unbounded `meilisearch-python-sdk`.
Use `AsyncClient` and `AsyncIndex` directly. Delete the `Connection` wrapper,
worker-thread dispatch, and locking. Use SDK `Hybrid` and `MeilisearchSettings`
models and typed search responses instead of manually handling API JSON.
Native settings dictionaries remain accepted, but fields unsupported by the SDK
are rejected explicitly to prevent its default silent field-dropping.

Stores created from connection configuration own
their client; stores receiving an existing client do not. `aclose()` closes only
owned clients. Collection setup closes its temporary client in `finally`, including
on errors and caller cancellation. Callers close shared clients themselves.

Keep a small task helper using the SDK's async `get_task`: SDK 7.5.2's waiter treats
only `succeeded` and `failed` as terminal, ignoring `canceled`. The helper handles
all three statuses, preserves task error details, and bounds polling and requests
with `asyncio.timeout`. It never cancels the server-side task. Index creation uses
the SDK's built-in wait with the configured timeout; cancellation there may surface
as a timeout. No custom HTTP transport or provider implementation is introduced.

## Consequences

This supersedes ADR-001's synchronous transport and ADR-003's SDK package name,
not the data layout, toolkit contracts, or unconstrained version policy. Existing
`meilisearch.Client` instances must be replaced with
`meilisearch_python_sdk.AsyncClient`. HTTP timeout units remain seconds; task timeout
units remain milliseconds. Config-created stores now need cleanup (for example,
`contextlib.aclosing`), because the async SDK owns a persistent HTTP client.

Hybrid/keyword search, facets, supplied `_vectors`, provider-generated embeddings,
navigation, patching, and write-before-delete replacement semantics are unchanged.

## Verification and sources

- Real-server tests: Meilisearch 1.53.2, SDK 7.5.2, toolkit 0.0.13.
- Regression tests cover owned/shared client cleanup, setup errors/cancellation,
  failed/canceled background tasks, timeout bounds, and failed replacement safety.
- [Async SDK source](https://github.com/sanders41/meilisearch-python-sdk)
- Context7: [AsyncClient docs](https://meilisearch-python-sdk.paulsanders.dev/async_client_api/)
  and [AsyncIndex docs](https://meilisearch-python-sdk.paulsanders.dev/async_index_api/)
