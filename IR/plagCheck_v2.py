import re
import math
import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class PlagiarismChecker:
    def __init__(self, database, alpha=0.85):
        self.database = database
        self.alpha = alpha
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def binary_distance(self, u, v):
        return 0 if u == v else 1

    def preprocess_document(self, document):
        title, content = document.split(":", 1)
        title = re.sub(r'[^a-zA-Z0-9 ]', '', title.lower())
        content = re.sub(r'[^a-zA-Z0-9 ]', '', content.lower())
        return title, content

    def stem_terms(self, terms):
        return [self.stemmer.stem(term) for term in terms if term not in self.stop_words]

    def calculate_tf_idf(self, document, total_documents, document_frequencies):
        terms = self.stem_terms(document.split())
        tf_idf = Counter()
        for term in set(terms):
            tf = terms.count(term) / len(terms)
            idf = math.log((total_documents + 1) / (0.5 + document_frequencies.get(term, 0.5)))
            tf_idf[term] = tf * idf
        return tf_idf

    def cosine_similarity(self, vector1, vector2):
        dot_product = sum(vector1[term] * vector2[term] for term in set(vector1) & set(vector2))
        magnitude1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
        magnitude2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 != 0 else 0

    def check_duplicate(self, new_document):
        new_title, new_content = self.preprocess_document(new_document)
        # A)verify titles
        for existing_document in self.database:
            existing_title, _ = self.preprocess_document(existing_document)
            if self.binary_distance(new_title, existing_title) == 0:
                print("Document is a duplicate. Discarding.")
                return

        # B) tfidf
        document_frequencies = Counter()
        for existing_document in self.database:
            _, existing_content = self.preprocess_document(existing_document)
            document_frequencies.update(set(self.stem_terms(existing_content.split())))

        new_vector = self.calculate_tf_idf(new_content, len(self.database), document_frequencies)

        # Print TF-IDF for new doc
        print("\nTF-IDF Table for New Document:")
        print("Term\t\tTF-IDF")
        for term, value in new_vector.items():
            print(f"{term}\t\t{value}")

        # C) similarity according to threshold
        plagiarized_words = []
        for existing_document in self.database:
            _, existing_content = self.preprocess_document(existing_document)
            existing_vector = self.calculate_tf_idf(existing_content, len(self.database), document_frequencies)
            similarity = self.cosine_similarity(new_vector, existing_vector)
            
            if similarity > self.alpha:
                print("Document is a duplicate. Discarding.")
                return
            else:
                # Check each word for plagiarism
                for term in new_vector.keys():
                    if term in existing_vector and self.cosine_similarity({term: new_vector[term]}, {term: existing_vector[term]}) > self.alpha:
                        plagiarized_words.append(term)

        if plagiarized_words:
            print("Document is plagiarized. Plagiarized words:")
            print(plagiarized_words)
        else:
            print("Document is not a duplicate. Adding to the database.")
            self.database.append(new_document)

            # Print TF-IDF table for old doc
            print("\nTF-IDF Table for Existing Documents:")
            for i, existing_document in enumerate(self.database, start=1):
                _, existing_content = self.preprocess_document(existing_document)
                existing_vector = self.calculate_tf_idf(existing_content, len(self.database), document_frequencies)
                print(f"\nDocument {i}:")
                print("Term\t\tTF-IDF")
                for term, value in existing_vector.items():
                    print(f"{term}\t\t{value}")

existing_documents = [
    "D1: information requirement: query considers the user feedback as information requirement to search.",
    "D2: information retrieval: query depends on the model of information retrieval used.",
    "D3: prediction problem: Many problems in information retrieval can be viewed as prediction problems",
    "D4: search: A search engine is one of applications of information retrieval models."
]

plagiarism_checker = PlagiarismChecker(existing_documents)

new_document = "D5: Feedback: feedback is typically used by the system to modify the query and improve prediction"
plagiarism_checker.check_duplicate(new_document)

new_document = "D6: information retrieval: ranking in information retrieval algorithms depends on user query"
plagiarism_checker.check_duplicate(new_document)
