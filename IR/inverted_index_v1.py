# Create an Inverted Index
#  Implement a program that reads through the list of sorted terms and creates an in-
# memory inverted index.
#  Treat the 10 most prominent words as stop words and let your program compute
# the index size after that change.
#  To process simple Boolean queries using the index file create above. 
#  Provide support for conjunctive queries and mixed operator queries.

class InvertedIndex:
    def __init__(self):
        self.inverted_index = {}

    def build_index(self, terms):
        for doc_id, term in enumerate(terms):
            if term not in self.inverted_index:
                self.inverted_index[term] = set()
            self.inverted_index[term].add(doc_id)

    def remove_stop_words(self, stop_words):
        for stop_word in stop_words:
            if stop_word in self.inverted_index:
                del self.inverted_index[stop_word]

    def process_query(self, query):
        terms = self.tokenize_query(query)
        result = set()

        for term in terms:
            if term in self.inverted_index:
                if not result:
                    result = self.inverted_index[term]
                else:
                    result = result.intersection(self.inverted_index[term])

        return result

    def tokenize_query(self, query):
        return query.lower().split()


def read_sorted_terms(file_path):
    with open(file_path, 'r') as file:
        terms = [line.strip() for line in file]

    return terms


file_path = "sorted_terms.txt" 
stop_words = ["the", "and", "is", "in", "of", "to", "it", "this", "that", "was"]

sorted_terms = read_sorted_terms(file_path)


index = InvertedIndex()
index.build_index(sorted_terms)

index.remove_stop_words(stop_words)

print("Inverted Index:")
for term, doc_ids in index.inverted_index.items():
    print(f"{term}: {doc_ids}")

query = "apple and orange"
query_result = index.process_query(query)
print(f"\nQuery Result for '{query}': {query_result}")

conjunctive_query = "apple and banana"
conjunctive_query_result = index.process_query(conjunctive_query)
print(f"\nQuery Result for '{conjunctive_query}': {conjunctive_query_result}")

mixed_query = "apple and (orange or banana) not kiwi"
mixed_query_result = index.process_query(mixed_query)
print(f"\nQuery Result for '{mixed_query}': {mixed_query_result}")
