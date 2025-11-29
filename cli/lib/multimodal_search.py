import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies, format_search_result, DEFAULT_SEARCH_LIMIT

class MultimodalSearch:
    def __init__(self, documents: list[dict]=None, model_name="clip-ViT-B-32"):
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image])  # type: ignore[arg-type]
        return image_embeddings[0]
    
    def search_with_image(self, image_path, limit=DEFAULT_SEARCH_LIMIT):
        image_embedding = self.embed_image(image_path)

        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in similarities[:limit]:
            doc = self.documents[idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:100],
                    score=score,
                )
            )

        return results


def verify_image_embedding(img_path):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path="data/paddington.jpeg", limit=5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    movies = load_movies()
    searcher = MultimodalSearch(movies)
    results = searcher.search_with_image(image_path, limit)

    return {"image_path": image_path, "results": results}