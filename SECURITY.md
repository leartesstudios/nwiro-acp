# Security Policy

## Supported versions

Security fixes are applied to the latest released version of `local-llm-acp`.
Please upgrade to the latest release before reporting an issue.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

- Use GitHub's **[Report a vulnerability](../../security/advisories/new)** (Security → Advisories), or
- email **support@leartesstudios.com**.

Include a description, the affected version or commit, and reproduction steps. We
aim to acknowledge within a few business days.

## Trust model

`local-llm-acp` is a local, single-tenant stdio shim. It is spawned as a child
process by a trusted host (the Nwiro UE5 Integration Kit) and pointed at a **local
or otherwise trusted** OpenAI-compatible LLM endpoint. It is designed for a
**trusted operator + trusted model** environment — not as a sandboxed, multi-tenant,
or internet-facing service.

- **Not a sandbox.** Tool calls are dispatched to the host's MCP server; high-impact
  tools (including any code-execution or filesystem tools the host exposes) are gated
  by the host's permission UI, not by the shim. Do not point the shim at an untrusted
  backend, and do not feed it input you would not run locally.
- **Network.** The shim only talks to the configured `NWIRO_LOCAL_LLM_BASE_URL`. It
  follows **no HTTP redirects** (anti-SSRF), so a backend cannot redirect the prompt,
  tool schemas, or bearer token to another host.
- **Secrets.** The API key is delivered only via `NWIRO_LOCAL_LLM_API_KEY_localllm`,
  is wrapped in a type whose `Debug` prints `[REDACTED]`, and is never logged or
  placed on the command line.
- **Resource bounds.** A runaway backend cannot hang or OOM the host: prompt rounds
  are bounded by a pre-stream timeout, streams by a per-token inactivity timeout and a
  wall-clock deadline, and the accumulated response by a hard size ceiling — each
  surfaces a clean, diagnosable refusal rather than a hang.

See the **Security notes** section of the [README](README.md) for the specific
hardening guarantees and the environment variables that tune these bounds.
