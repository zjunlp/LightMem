from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, tokenizer=None, k1=1.5, b=0.75):
        self.tokenizer = tokenizer or (lambda text: text.lower().split())
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.bm25 = None

    def index(self, corpus):
        self.corpus = corpus
        tokenized = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def retrieve(self, query, top_k=5):
        if self.bm25 is None:
            raise RuntimeError("BM25 not indexed. Call index() first.")
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.corpus[i], float(scores[i])) for i in ranked[:top_k]]

    def __repr__(self):
        return f"<BM25(k1={self.k1}, b={self.b}, docs={len(self.corpus)})>"
