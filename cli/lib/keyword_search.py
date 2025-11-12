from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        preprocessed_query = preprocess_text(query)
        if preprocessed_query in preprocessed_title:
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> str:
    text = text.lower()

    # str.translate() returns a copy of a string where characters have been replaced or removed based on a translation table
    # Static method str.maketrans() creates a translation table
    # string.punctuation is a string of ASCII characters which are considered punctuation characters in the C locale:
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
    # Empty translation table that simply eliminates the group of characters from the third argument
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text