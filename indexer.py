import os
import json
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from bm25 import BM25

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

CRAWLED_FOLDER = Path(os.path.abspath('')) / 'crawled/'
RESOURCE_FOLDER = Path(os.path.abspath('')) / 'resource/'
RESOURCE_FOLDER.mkdir(exist_ok=True)

class Preprocessor:
    def __init__(self, stop_dict, stem_cache):
        self.stop_dict = stop_dict
        self.stem_cache = stem_cache
        self.ps = PorterStemmer()

    def __call__(self, s):
        s = re.sub(r'[^A-Za-z]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = word_tokenize(s)
        s = [ss for ss in s if ss not in self.stop_dict]
        s = [word for word in s if len(word) > 2]
        s = [self.stem_cache[w] if w in self.stem_cache else self.ps.stem(w) for w in s]
        s = ' '.join(s)
        return s

def create_stem_cache(documents_df):
    tokenized_description = documents_df['text'].apply(lambda s: word_tokenize(s))
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)
    return stem_cache

class Pr:
    def __init__(self, alpha, crawled_folder):
        self.crawled_folder = crawled_folder
        self.alpha = alpha
    
    def url_extractor(self):
        url_maps = {}
        all_urls = set([])
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                all_urls.add(j['url'])
                for s in j.get('url_lists', []):
                    all_urls.add(s)
                url_maps[j['url']] = list(set(j.get('url_lists', [])))
        return url_maps, list(all_urls)
    
    def pr_calc(self):
        url_maps, all_urls = self.url_extractor()
        url_matrix = pd.DataFrame(columns=all_urls, index=all_urls)

        for url in url_maps:
            if len(url_maps[url]) > 0 and len(all_urls) > 0:
                url_matrix.loc[url] = (1 - self.alpha) * (1 / len(all_urls))
                url_matrix.loc[url, url_maps[url]] = url_matrix.loc[url, url_maps[url]] + (self.alpha * (1 / len(url_maps[url])))

        url_matrix.loc[url_matrix.isnull().all(axis=1), :] = (1 / len(all_urls))

        x0 = np.matrix([1 / len(all_urls)] * len(all_urls))
        P = np.asmatrix(url_matrix.values)

        prev_Px = x0
        Px = x0 * P
        i = 0
        while (any(abs(np.asarray(prev_Px).flatten() - np.asarray(Px).flatten()) > 1e-8)):
            i += 1
            prev_Px = Px
            Px = Px * P

        self.pr_result = pd.DataFrame(Px, columns=url_matrix.index, index=['score']).T.loc[list(url_maps.keys())]

class IndexerManual:
    def __init__(self, crawled_folder, preprocessor_obj):
        self.crawled_folder = crawled_folder
        self.preprocessor = preprocessor_obj
        self.stored_file = RESOURCE_FOLDER / 'manual_indexer.pkl'

    def run_indexer(self):
        documents = []
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                documents.append(j)
        self.documents = pd.DataFrame.from_dict(documents)
        
        tfidf_vectorizor = TfidfVectorizer(preprocessor=self.preprocessor, stop_words=stopwords.words('english'))
        self.bm25 = BM25(tfidf_vectorizor)
        self.bm25.fit(self.documents.apply(lambda s: ' '.join(s[['title', 'text']]), axis=1))
        
        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

if __name__ == '__main__':
    pr = Pr(alpha=0.85, crawled_folder=CRAWLED_FOLDER)
    pr.pr_calc()
    pr.pr_result.to_pickle(RESOURCE_FOLDER / 'pagerank_scores.pkl')

    # use data in crawled folder to create stem cache intead of job csv
    docs_temp = []
    for f in os.listdir(CRAWLED_FOLDER):
        if f.endswith(".txt"):
            docs_temp.append(json.load(open(os.path.join(CRAWLED_FOLDER, f))))
    df_temp = pd.DataFrame(docs_temp)
    
    stem_cache = create_stem_cache(df_temp)
    stop_dict = set(stopwords.words('english'))
    my_pre_processor = Preprocessor(stop_dict, stem_cache)

    indexer = IndexerManual(crawled_folder=CRAWLED_FOLDER, preprocessor_obj=my_pre_processor)
    indexer.run_indexer()
