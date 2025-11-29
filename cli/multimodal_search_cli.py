#! /usr/bin/env python3

import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verif_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verif_parser.add_argument("image", type=str, help="Path to image file")

    verif_parser = subparsers.add_parser(
        "image_search", help="Search documents using an image"
    )
    verif_parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            result = image_search_command(args.image)

            print(f"Image search results for: {result['image_path']}")
            print("=" * 60)

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
