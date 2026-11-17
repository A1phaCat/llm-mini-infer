# Security

## Reporting a vulnerability

Report a vulnerability through GitHub's private vulnerability reporting. Open the repository's Security tab and select "Report a vulnerability".

Do not report a vulnerability in a public issue, a pull request or a discussion.

Include these facts in the report:
- the version or image tag, and the settings that matter (`images.allow_urls`, whether an API key is set);
- the request, the configuration or the input that triggers the problem;
- what you expected, and what happened instead;
- the effect: what an attacker gains.

This project makes no promise about the time of a response.

## Supported versions

Only the latest release receives security fixes.

## The security model

llm-mini-infer is a service for a trusted network. It has one shared API key and no other access control. Anyone who can reach the server and holds the key can use every feature.

### The API key

The key comes from `LLM_MINI_INFER_API_KEY` or `VLLM_API_KEY`, never from a configuration file. With a key set, access is denied by default: every request needs `Authorization: Bearer <key>`, and so does a path the server does not serve. The open paths are OPTIONS, `/health`, `/ready`, `/live` and `/metrics`.

Without a key, every path is open. The start-up line warns when `server.host` is not a loopback address. See [Access](docs/operations.md#access).

### Image URLs

Image URLs are off by default (`images.allow_urls`). When on, the server fetches `http(s)` URLs and follows at most 5 redirects. A block refuses loopback, private, link-local, cloud-metadata and reserved addresses. The block checks each address after DNS resolution and on every redirect.

Two settings change the block:
- `images.allowed_networks` exempts CIDR ranges from it;
- `images.allowed_domains` limits fetching to the listed host names.

The block checks addresses, not what an address serves. It does not stop a request to a service on a public address, or to a network that `images.allowed_networks` exempts. It does not stop a route through a proxy that rewrites destinations. The status, timing and errors of a fetch from a public host still reach the client. See [Access](docs/operations.md#access) and [Settings](docs/operations.md#settings).

### Caches

`cache_salt` separates the prefix cache and the encoder cache. It is not a tenant boundary: all clients share the server, its queue and its GPU time.

### Limits

Each request has caps. A request over a cap gets a 400 before the server does the work, except the download total:
- the images: their number, their bytes, their pixels, and the total a request decodes; the total a request downloads stops the fetch once it is spent, with a 422 as in vLLM;
- the schema: `structured_output.max_schema_bytes`, and the structured-output caps on nesting, properties, alternatives, pattern text and fixed output.

A schema over a structured-output cap is a 400 before the grammar compiler runs. [Structured output](docs/vllm-compatibility.md#structured-output) lists the caps.

The queue caps `scheduler.max_num_queued_reqs` and `scheduler.max_num_queued_tokens` are off by default. Set them to refuse requests with a 503 when the server is full. See [Settings](docs/operations.md#settings).

## Out of scope

The server does not provide:
- TLS: terminate it in a proxy or a service mesh;
- multi-tenancy;
- rate limits;
- authentication beyond one key.

## Deployment advice

- Keep `/metrics` on a trusted network. It is open, and it names the model and counts traffic.
- Set an API key when `server.host` is not a loopback address.
- Prefer `images.allowed_domains` to `images.allowed_networks`.
