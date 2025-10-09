import re

from rank_bm25 import BM25Okapi

_token_re = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    return _token_re.findall((text or "").lower())


def score_bm25(query: str, docs: list[str]) -> list[float]:
    tok_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tok_docs)
    return list(map(float, bm25.get_scores(tokenize(query))))
