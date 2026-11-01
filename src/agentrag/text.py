"""Tokenisation shared by the lexical index and the hashing embedder."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its
    me my of on or our should so than that the their them there these they this to was we
    were what when where which who why will with you your""".split()
)


def stem(token: str) -> str:
    """A deliberately small suffix stripper.

    It is not Porter, but it folds the common English inflections
    (batteries/battery, charged/charging/charge) onto one form, which is
    what matters for lexical recall on short technical documents.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("sses"):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us")):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _WORD.findall(text.lower())
    return [stem(t) for t in tokens if not (drop_stopwords and t in STOPWORDS)]
