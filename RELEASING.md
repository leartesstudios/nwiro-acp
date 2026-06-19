# Releasing local-llm-acp

## Overview

`local-llm-acp` is the local LLM adapter control plane (ACP) shim. This document defines
the **release asset naming convention**. The filename convention is matched by the
host app's auto-update resolver — any deviation silently breaks updates.

---

## Filename Convention

Release assets must be named **exactly** as follows (example: tag `v0.1.0`):

| Platform       | Filename                                                      |
|----------------|---------------------------------------------------------------|
| Windows x64    | `local-llm-acp-0.1.0-x86_64-pc-windows-msvc.zip`           |
| Windows ARM64  | `local-llm-acp-0.1.0-aarch64-pc-windows-msvc.zip`          |
| macOS ARM64    | `local-llm-acp-0.1.0-aarch64-apple-darwin.tar.gz`          |
| macOS Intel    | `local-llm-acp-0.1.0-x86_64-apple-darwin.tar.gz`           |
| Linux x64      | `local-llm-acp-0.1.0-x86_64-unknown-linux-gnu.tar.gz`      |
| Linux ARM64    | `local-llm-acp-0.1.0-aarch64-unknown-linux-gnu.tar.gz`     |

**Pattern:** `local-llm-acp-<VERSION>-<TARGET>.<EXT>`

- `<VERSION>` is the git tag with the leading `v` stripped: `v0.1.0` becomes `0.1.0`
- `<TARGET>` is the Rust target triple (see table above)
- `<EXT>` is `zip` for Windows, `tar.gz` for macOS and Linux

The CI workflow derives `<VERSION>` from `$GITHUB_REF_NAME` by stripping the leading `v`:
- Bash: `${GITHUB_REF_NAME#v}`
- PowerShell: `$env:GITHUB_REF_NAME -replace '^v', ''`

---

## Archive Contents

Each archive contains **exactly one binary** at the top level:

- Windows: `local-llm-acp.exe`
- macOS / Linux: `local-llm-acp`

No nested directories. The C++ bridge unpacks the archive and looks for the binary at the archive
root. Archives are created by copying the binary to the workspace root first, then archiving that
single file — this guarantees a flat structure regardless of tooling.

---

## Target Triples

| Triple                         | Platform             | Runner           | Build method   |
|--------------------------------|----------------------|------------------|----------------|
| `x86_64-pc-windows-msvc`       | Windows x64          | `windows-latest` | Native cargo   |
| `aarch64-pc-windows-msvc`      | Windows ARM64        | `windows-11-arm` | Native cargo   |
| `aarch64-apple-darwin`         | macOS Apple Silicon  | `macos-14`       | Native cargo   |
| `x86_64-apple-darwin`          | macOS Intel          | `macos-14`       | Cross (native SDK) |
| `x86_64-unknown-linux-gnu`     | Linux x64            | `ubuntu-latest`  | cargo-zigbuild |
| `aarch64-unknown-linux-gnu`    | Linux ARM64          | `ubuntu-latest`  | cargo-zigbuild |

> **macOS Intel (`x86_64-apple-darwin`) was dropped in v0.1.13 and re-added** by
> CROSS-compiling on the `macos-14` (Apple Silicon) runner instead of the flaky
> `macos-13` Intel runner that caused the original removal. Both macOS binaries are
> pinned to `MACOSX_DEPLOYMENT_TARGET=11.0`, so they require macOS 11 (Big Sur)+
> (Intel Macs on Catalina 10.15 or earlier are not covered).

**Linux glibc floor:** glibc >= 2.28 (RHEL 8, Debian 10, Ubuntu 20.04 and later). Enforced via
the cargo-zigbuild target suffix syntax: `x86_64-unknown-linux-gnu.2.28`. Without the `.2.28`
suffix, zigbuild silently links against the host runner glibc (Ubuntu runners are typically
glibc 2.35+), producing binaries that fail on older distros with `GLIBC_2.35 not found`.

---

## Pre-Release Tags (WARNING)

The JS resolver uses the RegExp pattern `[\d.]+` to match the version segment in asset filenames.
This does **not** match pre-release suffixes.

| Tag           | Version in filename | JS RegExp matches? |
|---------------|---------------------|--------------------|
| `v0.1.0`      | `0.1.0`             | Yes                |
| `v0.1.0-rc.1` | `0.1.0-rc.1`        | No                 |

If you push a pre-release tag like `v0.1.0-rc.1`:
- CI will build and attach 5 assets with names like `local-llm-acp-0.1.0-rc.1-x86_64-...`
- The JS resolver will **silently fail to match** these assets
- Users will not receive the update

Before publishing a pre-release, update the host app's auto-update resolver RegExp
to support pre-release suffixes, for example:

```
[\d.]+(?:-[a-z0-9.]+)?
```

---

## Pre-Release Checklist

**Run BEFORE bumping `Cargo.toml` version or creating the git tag.** This
catches doc-rot, smoke regressions, and version-string mismatches before
they become a public release. Added in v0.1.22 after v0.1.21 shipped
with three stale README known-limitations entries.

- [ ] **README scan**: confirm `## Limitations` reflects current reality.
      If items were fixed in the upcoming release, move them to the
      "Previously resolved limitations" block in `CHANGELOG.md`.
- [ ] **Smoke suite**: run the test-harness scripts in `scripts/` against a
      local backend and confirm each prints its pass line.
- [ ] **Unit tests**: `cargo test --release` passes with zero failures.
- [ ] **Clean build**: `cargo build --release` produces no warnings.
- [ ] **Version match**: `Cargo.toml` `version` equals the intended
      git tag (without the `v` prefix). Mismatch is an automatic block —
      cross-platform CI uses `Cargo.toml` as source-of-truth. **Now enforced
      by the `verify-version` job in `release.yml`** (the release fails fast
      if tag ≠ `Cargo.toml` version), but check it here to avoid a wasted run.
- [ ] **Third-party NOTICE**: if dependencies changed, the release's
      `licenses` job regenerates `THIRD-PARTY-LICENSES.md` from `Cargo.lock`
      and `cargo deny check licenses` must pass. To preview locally:
      `cargo deny check licenses && cargo about generate about.hbs -o THIRD-PARTY-LICENSES.md`.
      A new dependency under a non-permissive license blocks the release until
      its SPDX id is reviewed and added to `deny.toml` + `about.toml`.
- [ ] **MODEL-SETUP.md cross-check**: if the release changes anything
      about backend configuration expectations (e.g. context length,
      probe behaviour, env vars), confirm `docs/MODEL-SETUP.md`
      matches.

## Release Checklist

- [ ] Bump `version` in `Cargo.toml` (e.g. `0.1.22`)
- [ ] Commit: `git commit -m "v0.1.22 — <one-line summary>"`
- [ ] Tag: `git tag v0.1.22`
- [ ] Push: `git push origin main && git push origin v0.1.22`
- [ ] **Within 60s, verify CI triggered**: `gh run list --limit 1` should
      show a `queued` or `in_progress` workflow run for the new tag.
      **If it didn't trigger** (known GitHub Actions deduplication quirk
      when commit + tag are pushed in quick succession to the same SHA),
      force-trigger with:
      ```
      git push --delete origin v0.1.22 && git push origin v0.1.22
      ```
      Deleting the remote tag first ensures the subsequent push registers
      as a fresh ref-creation event. Verified failure-then-recovery on
      v0.1.25.
- [ ] Wait for the GitHub Actions Release workflow (~10 min)
- [ ] Verify all 5 platform archives **plus `THIRD-PARTY-LICENSES.md`** (the
      attribution NOTICE attached by the `licenses` job) are on the release page
- [ ] Verify each asset filename exactly matches the Filename Convention table above
- [ ] Confirm each asset filename matches the Filename Convention table above
- [ ] Download each asset and verify the archive contains exactly one binary at the top level
- [ ] Verify each binary runs on a matching host (e.g. `./local-llm-acp --version`)

---

## License

Apache-2.0. See `LICENSE` in the shim repository.
