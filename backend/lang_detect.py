"""Local language detection for routing — library-based, no hardcoded rules.

Replaces the per-request LLM language-classification round-trip on the critical
path. Uses the ``lingua`` detector (purpose-built for accuracy on short text),
constrained to the languages this assistant serves. Returns a 2-letter ISO
639-1 code, or ``"unknown"`` when the detector has no confident answer
(callers default unknown -> Spanish).

The served-language set is configurable via the ``LANG_DETECT_LANGUAGES`` env
var (comma-separated ISO 639-1 codes); it defaults to the same set the system
prompt supports. The detector is built once and cached.
"""
from __future__ import annotations

import os
from functools import lru_cache

# Default served languages — mirrors build_tool_system_prompt's language map.
_DEFAULT_LANGS = "en,es,fr,pt,it,de,hi"


def _wanted_codes() -> list[str]:
    raw = (os.getenv("LANG_DETECT_LANGUAGES") or _DEFAULT_LANGS)
    codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
    return codes or _DEFAULT_LANGS.split(",")


@lru_cache(maxsize=1)
def _detector():
    """Build (once) a lingua detector constrained to the served languages."""
    from lingua import Language, LanguageDetectorBuilder

    by_iso = {lang.iso_code_639_1.name.lower(): lang for lang in Language.all()}
    wanted = [by_iso[c] for c in _wanted_codes() if c in by_iso]
    if len(wanted) < 2:
        # lingua requires >= 2 languages; fall back to the full default set.
        wanted = [by_iso[c] for c in _DEFAULT_LANGS.split(",") if c in by_iso]
    return LanguageDetectorBuilder.from_languages(*wanted).build()


def warmup() -> None:
    """Preload the detector models so the first real request isn't slow."""
    try:
        _detector().detect_language_of("warmup")
    except Exception:
        pass


def detect_language(text: str) -> str:
    """Return a 2-letter ISO 639-1 language code, or 'unknown'."""
    text = (text or "").strip()
    if not text:
        return "unknown"
    try:
        result = _detector().detect_language_of(text)
    except Exception:
        return "unknown"
    if result is None:
        return "unknown"
    return result.iso_code_639_1.name.lower()
