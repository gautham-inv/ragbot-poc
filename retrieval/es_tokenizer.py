from __future__ import annotations

import re


_token_re = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)


def tokenize_es(text: str) -> list[str]:
    # Lowercase, keep alphanumerics, preserve product codes like GL00369.
    return [t.lower() for t in _token_re.findall(text)]

