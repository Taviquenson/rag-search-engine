import os, pickle
from .search_utils import load_movies, PROJECT_ROOT

class InvertedIndex:
    # Class attributes (shared by all instances)
    index = {} # maps tokens (strings) to sets of document IDs (integers)
    docmap = {} # maps document IDs to their full document objects

    # def __init__(self):
    #     # Constructor method: initializes instance attributes
    #     self.index = {}
    #     self.docmap = {}

    def __add_document(self, doc_id, text): #private method
        # Tokenize the input text
        text = text.lower()
        tokens = text.split()
        # Add each token to the index with the document ID
        for token in tokens:
            if token: # making sure the token is valid/truthy
                if token in self.index:
                    self.index[token].add(doc_id)
                else:
                    self.index[token] = {doc_id}

    def get_documents(self, term):
        # get the set of document IDs for a given token
        doc_ids = self.index[term]
        # return them as a list, sorted in ascending order
        return sorted(doc_ids)

    def build(self):
        movies = load_movies()
        # Iterate over all the movies and add them to both the index and the docmap
        for m in movies:
            # adding movie to the index
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            # adding movie to docmap
            self.docmap[m["id"]] = m

    def save(self):
        directory_path = os.path.join(PROJECT_ROOT, "cache")
        # Create the cache directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # save the index and docmap attributes to disk using the pickle module's dump function
        # Open a file in binary write mode ('wb')
        index_path = os.path.join(directory_path, "index.pkl")
        docmap_path = os.path.join(directory_path, "docmap.pkl")
        # a with statement ensures the file is automatically closed
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
