import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    DOCUMENT_PREVIEW_LENGTH,
    load_movies,
    load_stopwords,
    format_search_result,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set) # Is like a dict but auto-creates a default value when a missing key is accessed
                                      # Here, each new key gets a set() by default.
                                      # Maps a term to a set of document IDs
        self.docmap: dict[int, dict] = {} # dict of document IDs to movie objects
        self.term_frequencies = defaultdict(Counter) # Counter is a dictionary optimized for counting
        self.doc_lengths = defaultdict(int)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
    
    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set()) # Here, if term isn’t in the index, you get an empty set() instead of a KeyError.
                                              # Although with defaultdict(set): doc_ids = self.index[term] already gives a set or creates one
                                              # And if using a plain dict: doc_ids = self.index[term] if term in self.index else set()
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        # cnt = Counter()
        # for token in tokens:
        #     cnt[token] += 1
        # self.term_frequencies[doc_id] = cnt
        # Since Counter is built for routines like the above 4 lines of code
        # The same behaviour can be achieved with just this line of code:
        self.term_frequencies[doc_id].update(tokens) # Works like dict.update() but adds counts instead of replacing them
        self.doc_lengths[doc_id] = len(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        # Tokenize the term, but assume that there is only one token.
        tokens = tokenize_text(term)
        if len(tokens) != 1:  # If there's more than one, raise an exception.
            raise Exception("term must be a single token")
        # return the times the token appears in the document with the given ID
        # If the term doesn't exist in that document, return 0
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
        # The term_frequencies definition in the constructor auto-creates a default value when a missing key is accessed.
        # I guess by default the behaviour below where 0 is returned if term isn't in the Counter dict is implemented as well via defaultdict(Counter)
        # return self.term_frequencies[doc_id].get(token, 0)

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1: 
            raise Exception("term must be a single token")
        token = tokens[0]
        # Calculate the IDF for the given term
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise Exception("term must be a single token")
        token = tokens[0]
        # Calculate the BM245 IDF for the given term
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) +1)

    def get_bm25_tf(self, doc_id: int, term: str, k1 = BM25_K1, b = BM25_B):
        # Length normalization factor
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        # Apply to term frequency
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_docs_length = 0
        for doc_length in self.doc_lengths.values(): # REMEMBER, without .values() the keys are returned!
            total_docs_length += doc_length
        return float(total_docs_length / len(self.doc_lengths))
    
    def bm25(self, doc_id: int, term: str) -> float:
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query: str, limit: int) -> list[dict]:
        query_tokens = tokenize_text(query)

        scores = {}  # maps document IDs to their total BM25 scores
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
            
        # Sort the scores dict in descending order into a list of tuples
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Return the top limit documents
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"][:DOCUMENT_PREVIEW_LENGTH]+"...",
                score=score,
            )
            results.append(formatted_result)

        return results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:  # avoids having repeats of movie results
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1 = BM25_K1, b = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b) 


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
