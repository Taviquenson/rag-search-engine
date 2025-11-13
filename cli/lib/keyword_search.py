from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        preprocessed_query = preprocess_text(query)
        # preprocess_text returns a list of tokens

        if check_partial_match(preprocessed_query, preprocessed_title):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> list[str]:
    # Text preprocessing pipeline

    # Case Sensitivity
    text = text.lower()

    # Punctuation
    # str.translate() returns a copy of a string where characters have been replaced or removed based on a translation table
    # Static method str.maketrans() creates a translation table
    # string.punctuation is a string of ASCII characters which are considered punctuation characters in the C locale:
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
    # Empty translation table that simply eliminates the group of characters from the third argument
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization: so we can search for partial matches, not just exact substrings
    # i.e. the query "Great Bear" will match the title "Big Bear" because the "bear" token is present in both
    tokens = text.split()
    # When called without arguments, str.split() automatically:
    # - Treats any sequence of whitespace characters (spaces, tabs, newlines) as a single delimiter.
    # - Ignores leading and trailing whitespace entirely.

    return tokens

# This complements the Tokenization step preprocessing step
# Using nested loops and the `in`` operator, combined with the any() function, it stops checking as soon as a match is found.
def check_partial_match(list_Q, list_T):
    """
    Checks if any token in list_Q is a substring of any token in list_T.
    """
    # Iterate through every query token
    for q_token in list_Q:
        # Check if this query token is in any of the target tokens
        # `any()` stops as soon as it finds the first True value (the match)
        if any(q_token in t_token for t_token in list_T): # any() is being provided a "generator expression"
                                                          # which looks similar to a list comprehension
            # This generator expresion acts like a very efficient loop that creates a sequence of True or False values (Booleans) on the fly.
            # for t_token in list_T: iterates through every single word in list_T (the target list).
            # q_token in t_token: For each iteration, this expression performs a substring check: "Is the current q_token a part of the current t_token?".
            
            # i.e. If list_T were ['banana', 'faster', 'quickest'] and q_token was 'fast', the generator produces this sequence of values internally:
            # 'fast' in 'banana' -> False
            # 'fast' in 'faster' -> True
            # 'fast' in 'quickest' -> False

        # The any() function accepts an iterable (like the sequence of True/False values generated in the example above)
        # and determines if at least one item in that sequence is True.
        # Here is the crucial behavior of any():
        # - It Short-Circuits: any() is "lazy." As soon as it encounters the very first True value in the sequence, it stops iterating immediately and returns True. It does not waste time checking the rest of the items.
        # - If it goes through the entire sequence without finding a single True value, it returns False.
            
            # A match was found, we can return True immediately
            return True
    
    # If the loops finishes without finding any match
    return False