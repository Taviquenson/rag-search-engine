from dotenv import load_dotenv
from google import genai
import os
import json

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def llm_evaluation(query: str, documents: list[dict], limit: int = 5):
    if not documents:
        return []
    
    doc_map = {}
    doc_list = []
    for i, doc in enumerate(documents,1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{i}. {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    # formatted_results = "\n".join(doc_list)

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(doc_list)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    
    response = client.models.generate_content(model=model, contents=prompt)
    ratings_text = (response.text or "").strip()

    # If the model wrapped it in ```json ...``` or ``` ...```, strip that off
    if ratings_text.startswith("```"):
        # Remove the first opening fence line
        lines = ratings_text.splitlines()
        # Drop the first line (``` or ```json) and, if present, the last line (```)
        if lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        ratings_text = "\n".join(lines).strip()

    parsed_ratings = json.loads(str(ratings_text))
    
    evals = []
    for doc, rating in zip(documents, parsed_ratings):
        evals.append(
            {
                "title": doc.get('title', ''),
                "rating": rating,
             })
        
    return evals

