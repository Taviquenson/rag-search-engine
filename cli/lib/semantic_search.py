import os
from sentence_transformers import SentenceTransformer
import numpy as np

from .search_utils import CACHE_DIR, load_movies

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    # generates an embedding for a single text input, and verify that it works
    def get_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")   
        return self.model.encode([text])[0] # only care about first list element because only passing in one input

    def build_embeddings(self, documents):
        self.documents = documents # documents is a list of dictionaries, each representing a movie
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        # Save the embeddings in a .npy file
        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents # documents is a list of dictionaries, each representing a movie
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.get_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")