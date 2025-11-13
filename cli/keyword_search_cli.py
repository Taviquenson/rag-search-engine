#!/usr/bin/env python3

import argparse
import json

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index for faster searches")
    search_parser.add_argument("query", type=str, help="Build query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for: " + args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building the Inverted Index and saving it to disk")
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            docs = inverted_index.get_documents('merida')
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()