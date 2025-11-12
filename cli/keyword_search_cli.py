#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for: " + args.query)
            try:
                with open("./data/movies.json", 'r') as file:
                    searchDict = json.load(file)
                results = [] # list of movie dicts
                for movie in searchDict["movies"]:
                    if (args.query) in movie["title"]:
                        results.append(movie)
                results.sort(key=lambda movie: movie['id']) #sorts in ascending order by id
                for i in range(0,5):
                    print(f"{str(i+1)}. {results[i]["title"]}")
            except FileNotFoundError:
                print("Error: 'config.json' not found.")
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in 'config.json'.")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()