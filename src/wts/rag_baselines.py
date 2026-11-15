"""RAG defenses compared against WTS in E2.

paraphrase_rerank   PoisonedRAG [14] evaluates query paraphrasing; we paraphrase with the backend LLM and
                    re-rank the same top-100 candidates + adversarial passages with the paraphrased query.
                    (Full-corpus re-retrieval is not repeated; this is stated in the paper.)
robustrag_keyword   Xiang et al. [19], keyword aggregation, re-implemented from the paper's description:
                    answer each passage in isolation, extract keywords, keep keywords that appear in at least
                    max(alpha * k, beta) isolated answers, answer from the retained keywords only.
                    Before reporting, check agreement with the authors' implementation on 20 questions and
                    record it in docs/status.md (genuineness rule: re-implementations are validated).
Perplexity filtering of chunks uses wts.baselines.PerplexityScorer with a conformal threshold.
"""
from __future__ import annotations

import re
from collections import Counter

PARAPHRASE_PROMPT = ("This is my question: {q}\nPlease craft 1 paraphrased version for the question.\n"
                     "Give your reply as a single line with only the paraphrased question.")

ISOLATED_PROMPT = ("Answer the question given only the information in the context. Answer in a few words. "
                   "If the context does not contain the answer, say \"I don't know\".\n\n"
                   "Context: {c}\n\nQuestion: {q}\n\nAnswer:")

KEYWORD_PROMPT = ("Answer the question using the following keywords, which were extracted from several "
                  "independently retrieved passages. Answer in a few words. If the keywords are not enough, "
                  "say \"I don't know\".\n\nKeywords: {k}\n\nQuestion: {q}\n\nAnswer:")

_STOP = set("""a an the of to in on for and or is are was were be been being by with as at from that this these those
it its it's i you he she they we not no yes i don't know dont unknown context answer question""".split())


def keywords(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    return {t for t in toks if t not in _STOP and len(t) > 1}


def robustrag_keyword(llm, question: str, contexts: list[str], alpha=0.3, beta=3.0) -> tuple[str, dict]:
    answers = [llm.ask(ISOLATED_PROMPT.format(c=c, q=question), max_tokens=32) for c in contexts]
    counts = Counter()
    for a in answers:
        if "i don't know" in a.lower():
            continue
        counts.update(keywords(a))
    thr = max(alpha * len(contexts), beta)
    kept = sorted([w for w, n in counts.items() if n >= thr], key=lambda w: -counts[w])
    if not kept:
        return "I don't know", dict(isolated=answers, kept=[])
    return llm.ask(KEYWORD_PROMPT.format(k=", ".join(kept), q=question), max_tokens=32), dict(isolated=answers, kept=kept)


def paraphrase(llm, question: str) -> str:
    out = llm.ask(PARAPHRASE_PROMPT.format(q=question), max_tokens=64).strip().splitlines()
    return out[0].strip() if out and out[0].strip() else question
