
import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    # Step 1: Tokenize and build vocabulary
    tokenized = [doc.lower().split() for doc in documents]
    vocab = sorted(set(word for doc in tokenized for word in doc))
    word_index = {word: i for i, word in enumerate(vocab)}
    
    N = len(documents)
    matrix = np.zeros((N, len(vocab)), dtype=float)
    
    for doc_idx, tokens in enumerate(tokenized):
        token_count = Counter(tokens)
        total_terms = len(tokens)
        
        # Step 2: TF = count(t,d) / total terms in d
        for word, count in token_count.items():
            tf = count / total_terms
            
            # Step 3: IDF = log(N / df(t))
            df = sum(1 for doc in tokenized if word in doc)
            idf = math.log(N / df)
            
            # Step 4: TF-IDF = TF * IDF
            matrix[doc_idx][word_index[word]] = tf * idf
    
    return matrix, vocab