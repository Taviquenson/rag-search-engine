from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # generates an embedding for a single text input, and verify that it works
    def get_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")   
        return self.model.encode([text])[0] # only care about first list element because only passing in one input


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.get_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")