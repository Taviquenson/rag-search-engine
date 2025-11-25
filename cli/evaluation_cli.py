import argparse

from lib.hybrid_search import HybridSearch
from lib.search_utils import (
    load_movies,
    load_golden_dataset,
    RRF_K,
)

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    test_cases = load_golden_dataset()

    searcher = HybridSearch(load_movies())
    print(f"k={args.limit}\n")
    for case in test_cases:
        top_k = searcher.rrf_search(case["query"], RRF_K, args.limit)
        # Calculate Precision@K
        relevant = []
        retrieved = []
        for res in top_k:
            retrieved.append(res["title"])
            if res["title"] in case["relevant_docs"]:
                relevant.append(res["title"])
        print(f"- Query: {case["query"]}")
        print(f"  - Precision@{args.limit}: {(len(relevant) / len(retrieved)):.4f}")
        print(f"  - Retrieved: {", ".join(retrieved)}")
        print(f"  - Relevant: {", ".join(relevant)}")
        print()


if __name__ == "__main__":
    main()