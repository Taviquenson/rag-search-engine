import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np

from .search_utils import(
 CACHE_DIR,
 DEFAULT_CHUNK_OVERLAP,
 DEFAULT_CHUNK_SIZE,
 DEFAULT_SEARCH_LIMIT,
 DEFAULT_SEMANTIC_CHUNK_SIZE,
 load_movies
)

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    # generates an embedding for a single text input, and verify that it works
    def generate_embedding(self, text):
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
    
    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded loaded. Call `load_or_create_embeddings` first.")

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        # Create a list of (similarity_score, document) tuples
        scores = []
        for i, embedding in enumerate(self.embeddings):
            scores.append( (cosine_similarity(embedding, query_embedding), self.documents[i]) )
        # Sort the list by similarity score in descending order
        scores.sort(key=lambda item: item[0], reverse=True)
        # Return the top results (up to limit) as a list of dictionaries
        results = []
        for score in scores[:limit]:
            results.append({
                "score": score[0],
                "title": score[1]["title"],
                "description": score[1]["description"]
            })
        return results


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
    
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
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['description'][:100]}..." )
        print()

def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    words = text.split()
    chunks = []    
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        # If `chunks` is truthy (which it is we've already added at least one chunk)
        # this prevents breaking on the very first iteration
        # AND
        # If there aren't words left after this chunk
        if chunks and len(chunk_words) <= overlap: # avoids a redundant or empty final chunk
            break # because otherwise you would either append ONLY overlap words
                  # OR append an empty chunk
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text) 
    # The regex splits on whitespace following a period, question mark, or exclamation point.
    # (?<=[.!?]) is a positive lookbehind assertion, ensuring the punctuation is kept.
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap: # avoids a redundant or empty final chunk
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks

def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")