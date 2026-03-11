import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):

    N = len(docs)
    doc_lens = [len(doc) for doc in docs]
    avgdl = sum(doc_lens) / N if N > 0 else 0

    # compute document frequency
    df = Counter()
    for doc in docs:
        unique = set(doc)
        for term in query_tokens:
            if term in unique:
                df[term] += 1

    scores = []

    for doc, dl in zip(docs, doc_lens):
        tf = Counter(doc)
        score = 0

        for term in query_tokens:
            term_freq = tf[term]
            if term_freq == 0:
                continue

            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

            denom = term_freq + k1 * (1 - b + b * dl / avgdl)

            if denom == 0:
                continue

            score += idf * ((term_freq * (k1 + 1)) / denom)

        scores.append(score)

    return np.array(scores)