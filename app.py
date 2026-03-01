import os
import re
import time
import pickle
from pathlib import Path
import pandas as pd
from flask import Flask, request, render_template
from elasticsearch import Elasticsearch
from bm25 import BM25
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

load_dotenv()

es_user = os.getenv("ES_USER")
es_pass = os.getenv("ES_PASSWORD")

app = Flask(__name__)

app.es_client = Elasticsearch("https://localhost:9200", basic_auth=(
    es_user, es_pass), ca_certs="http_ca.crt")

app.pr_result = pd.read_pickle('resource/pagerank_scores.pkl')

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
        s = [self.stem_cache[w]
             if w in self.stem_cache else self.ps.stem(w) for w in s]
        s = ' '.join(s)
        return s

class IndexerManual:
    def __init__(self):
        self.stored_file = 'resource/manual_indexer.pkl'
        with open(self.stored_file, 'rb') as f:
            cached_dict = pickle.load(f)
        self.__dict__.update(cached_dict)

    def query(self, q):
        return_score_list = self.bm25.transform(q)
        hit = (return_score_list > 0).sum()
        rank = return_score_list.argsort()[::-1][:hit]
        results = self.documents.iloc[rank].copy().reset_index(drop=True)
        results['score'] = return_score_list[rank]
        return results

app.manual_indexer = IndexerManual()

# I can't find a way to detect 2-3 sentence around the query since it's in Thai so i decide to just get 150 characters before and after the first match instead
def generate_snippet(text, query):
    if isinstance(text, list):
        text = " ".join(text)
    if not isinstance(text, str) or not text.strip():
        return ""

    match = re.search(re.escape(query), text, re.IGNORECASE)

    if match:
        start = max(0, match.start() - 150)
        end = min(len(text), match.end() + 150)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
    else:
        snippet = text[:300] + "..."

    # Surround the query term with <b> Note: It's really bad way to handle it i know
    highlighted_snippet = re.sub(
        f"({re.escape(query)})", 
        r"<b>\1</b>", 
        snippet, 
        flags=re.IGNORECASE
    )
    
    return highlighted_snippet

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/search_es_pr', methods=['GET'])
def search_es_pr():
    start = time.time()
    response_object = {'status': 'success'}
    arg_list = request.args.to_dict(flat=False)
    query_term = arg_list['query'][0]
    
    results = app.es_client.search(
        index='simple', 
        source_excludes=['url_lists'], 
        size=100,
        query={
            "script_score": {
                "query": {"match": {"text": query_term}}, 
                "script": {"source": "_score * doc['pagerank'].value"}
            }
        }
    )
    
    end = time.time()
    total_hit = results['hits']['total']['value']
    
    results_df = pd.DataFrame([
        [
            hit["_source"]['title'], 
            hit["_source"]['url'], 
            generate_snippet(hit["_source"]['text'], query_term), 
            hit["_score"]
        ]
        for hit in results['hits']['hits']
    ], columns=['title', 'url', 'text', 'score'])

    response_object['total_hit'] = total_hit
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start

    return response_object

@app.route('/search_manual_pr', methods=['GET'])
def search_manual_pr():
    start = time.time()
    response_object = {'status': 'success'}
    arg_list = request.args.to_dict(flat=False)
    query_term = arg_list['query'][0]

    results = app.manual_indexer.query(query_term)
    
    # Calculate score defult to 0 if result is not found
    if not results.empty and not app.pr_result.empty:
        results = results.merge(app.pr_result, left_on='url', right_index=True)
        results['total_score'] = results['score_x'] * results['score_y']
        results = results.sort_values(by='total_score', ascending=False)
    else:
        results['total_score'] = 0

    end = time.time()
    total_hit = len(results)
    
    results_df = pd.DataFrame([
        [
            row['title'],
            row.get('url', None),
            generate_snippet(row['text'], query_term),
            row['total_score']
        ]
        for _, row in results.iterrows()
    ], columns=['title', 'url', 'text', 'total_score'])

    response_object['total_hit'] = total_hit
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start

    return response_object

if __name__ == '__main__':
    app.run(port=5000, debug=True)
