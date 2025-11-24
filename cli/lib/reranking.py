import json
import os
from time import sleep

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def llm_rerank_individual(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        
        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip().strip('"')
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)
    # return results sort by the new score in descending order
    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def llm_rerank_batch(query: str, documents: list[dict], doc_list_str: str, limit: int = 5) -> list[dict]:
    scored_docs = []
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(model=model, contents=prompt)
    ranking_text = (response.text or "").strip().strip("'")

    # If the model wrapped it in ```json ...``` or ``` ...```, strip that off
    if ranking_text.startswith("```"):
        # Remove the first opening fence line
        lines = ranking_text.splitlines()
        # Drop the first line (``` or ```json) and, if present, the last line (```)
        if lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        ranking_text = "\n".join(lines).strip()

    ids = json.loads(str(ranking_text))
    for i, id in enumerate(ids,1):
        for doc in documents:
            if doc["id"] == id:
                scored_docs.append({**doc, "batch_rank": i})
    # return results sort by their batch_rank key from first to last
    scored_docs.sort(key=lambda x: x["batch_rank"])
    return scored_docs[:limit]


def rerank(query: str, documents: list[dict], method: str = "batch", limit: int = 5) -> list[dict]:
    match method:
        case "individual":
            return llm_rerank_individual(query, documents, limit)
        case "batch":
            doc_list_str = ""
            for i, doc in enumerate(documents, 1):
                doc_list_str += f"\n{i}. [id: {doc['id']}] {doc['title']} - {doc['document']}"
            return llm_rerank_batch(query, documents, doc_list_str, limit)
        case _:
            return documents[:limit]
