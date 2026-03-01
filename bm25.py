from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()


class BM25(object):
    def __init__(self, vectorizer, b=0.75, k1=1.6):
        self.vectorizer = vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        self.vectorizer.fit(X)
        self.y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        b, k1, avdl = self.b, self.k1, self.avdl

        len_y = self.y.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        y = self.y.tocsc()[:, q.indices]
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        idf = self.vectorizer.idf_[q.indices] - 1.
        numer = y.multiply(np.broadcast_to(idf, y.shape)) * (k1 + 1)

        return (numer / denom).sum(1).A1
