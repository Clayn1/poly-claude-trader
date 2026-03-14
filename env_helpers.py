import os


# ── Env helpers ────────────────────────────────────────────────────────────────

def _clean(raw: str) -> str:
    """Strip inline comments and surrounding whitespace from an env value.

    dotenv passes comments through verbatim when values aren't quoted:
        BANKROLL=100   # comment  →  os.getenv returns '100   # comment'
    This causes float/int parsing to fail and bool checks to silently
    mis-evaluate (e.g. DRY_RUN=true # note  →  treated as False → LIVE).
    """
    # Split on first ' #' or '	#' that looks like an inline comment
    for sep in (" #", "	#"):
        if sep in raw:
            raw = raw[:raw.index(sep)]
    return raw.strip()


def env_float(key: str, default: float) -> float:
    try:
        return float(_clean(os.getenv(key, str(default))))
    except ValueError:
        return default


def env_int(key: str, default: int) -> int:
    try:
        return int(_clean(os.getenv(key, str(default))))
    except ValueError:
        return default


def env_bool(key: str, default: bool) -> bool:
    val = _clean(os.getenv(key, str(default))).lower()
    return val in ("true", "1", "yes")


def env_str(key: str, default: str = "") -> str:
    return _clean(os.getenv(key, default))