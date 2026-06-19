"""Schema-bleed oracle — a 1:1 Python port of the shim's detector.

Mirrors `src/openai/client.rs::looks_like_schema_bleed` (v0.1.27) EXACTLY so the
model test harness (`scripts/tool-curve.py` Layer 1 + `scripts/model-matrix.py`
Layer 2) keys off the *same* definition of "bleed" the shim uses. Both layers
import `looks_like_schema_bleed` from here.

If the Rust detector changes, update this port AND re-run the fidelity test
(`python scripts/_bleed_oracle.py`) — it asserts byte-for-byte agreement on the
canonical Rust fixtures, including the GLM-4.5-air regression anchor
(`client.rs::schema_bleed_detects_glm_user_sample`).

Port notes:
- Python `str` is already a sequence of Unicode code points, so `len(s)` matches
  Rust `content.chars().count()` (the v0.1.27 critic-FINDING-4 scalar-not-byte fix).
- `str.count(x)` is non-overlapping, matching Rust `content.matches(x).count()`.
- `*100 // char_count` is integer division, matching Rust `*100 / char_count`.
"""

# The 8 structural characters from client.rs:182 — " : { } [ ] , and space.
_STRUCTURAL = set('":{}[], ')


def looks_like_schema_bleed(content: str) -> bool:
    """True when `content` looks like the model echoed the tool JSON schema as
    literal text. All three gates must hold (client.rs:164-187)."""
    char_count = len(content)  # Unicode scalars == Rust chars().count()
    if char_count < 50:  # gate 1: filter short prose
        return False
    schema_keyword_total = (
        content.count("object") + content.count('"type"') + content.count("properties")
    )
    if schema_keyword_total < 5:  # gate 2: needs schema vocabulary
        return False
    structural = sum(1 for c in content if c in _STRUCTURAL)
    return structural * 100 // char_count > 50  # gate 3: >50% structural


# --------------------------------------------------------------------------- #
# Fidelity test — `python scripts/_bleed_oracle.py`
# Ported from the Rust unit tests in client.rs (the inputs are verbatim).
# --------------------------------------------------------------------------- #

# Verbatim GLM-4.5-air bleed sample — client.rs:1299-1306
# (`schema_bleed_detects_glm_user_sample`, the v0.1.27 regression anchor).
_GLM_BLEED = ''':
        " " " "object: "object
        "":" " : object
        " : "object",
        " ":"s": " "object : "object "object" ] "object" "object": " },
        object " : "object } object": "object": " : "": "object":
        "object": "object" : "object": "object", "type": "object",
        "properties": { "type": "object" }, "object": "object'''

# Clears gate 2 (>=5 schema keywords) but is mostly prose -> gate 3 fails.
# (Mirrors client.rs `with_prose` fixture.)
_KEYWORD_PROSE = (
    "To use this tool you describe an object with a type field and several "
    "properties; the object holds an object and each object has a type and "
    "more properties than a plain object would."
)

_FIXTURES = [
    # (name, content, expected)
    ("glm_user_sample (regression anchor)", _GLM_BLEED, True),
    ("empty_string", "", False),
    ("short_string", '{"type":"object"}', False),  # < 50 chars -> gate 1
    (
        "normal_prose",
        "Sure! I can spawn a point light at the origin for you using the tool.",
        False,
    ),  # >= 50 chars, no schema vocabulary -> gate 2
    ("keyword_prose_low_structural", _KEYWORD_PROSE, False),  # gate 2 pass, gate 3 fail
]


def _main() -> int:
    failures = 0
    for name, content, expected in _FIXTURES:
        got = looks_like_schema_bleed(content)
        ok = got == expected
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected={expected} got={got}")
    if failures:
        print(f"\nFIDELITY FAIL: {failures} fixture(s) diverged from the Rust oracle.")
        return 1
    print(f"\nFIDELITY OK: {len(_FIXTURES)} fixtures match src/openai/client.rs::looks_like_schema_bleed.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
