import math
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def binary_tf(t, d):
    return int(t in d.split())

def binary_idf(t):
    N = 0
    for doc in db:
        if t in doc.split(':')[1].split():
            N += 1
    return math.log((len(db) + 1) / (N + 0.5))

def binary_tf_idf(t, d):
    return binary_tf(t, d) * binary_idf(t)

def rocchio_algorithm(query, relevant_docs, irrelevant_docs, alpha=1, beta=0.75, gamma=0.15):
    relevant_vectors = np.array([documentVectorMatrix[dbDocTitles.index(doc)] for doc in relevant_docs])
    irrelevant_vectors = np.array([documentVectorMatrix[dbDocTitles.index(doc)] for doc in irrelevant_docs])

    relevant_mean = np.sum(relevant_vectors, axis=0) / len(relevant_vectors) if relevant_vectors.any() else np.zeros(len(query))
    irrelevant_mean = np.sum(irrelevant_vectors, axis=0) / len(irrelevant_vectors) if irrelevant_vectors.any() else np.zeros(len(query))

    new_query = [str(alpha * int(q in query) + beta * relevant_mean[i] - gamma * irrelevant_mean[i]) for i, q in enumerate(all_terms)]
    return new_query



db = [
    'information requirement:query considers the user feedback as information requirement to search',
    'information retrieval:query depends on the model of information retrieval used',
    'prediction problem:Many problems in information retrieval can be viewed as prediction problems',
    'search:A search engine is one of applications of information retrieval models'
]

docsToAdd = [
    'Feedback:feedback is typically used by the system to modify the query and improve prediction',
    'information retrieval:ranking in information retrieval algorithms depends on user query',
]

dbDocTitles = [doc.split(':')[0] for doc in db]
print("dbDocTitles:", dbDocTitles)

# Add new documents to the database
for doc in docsToAdd:
    if doc.split(':')[0] not in dbDocTitles:
        db.append(doc)
        dbDocTitles.append(doc.split(':')[0])


stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
processed_docs = {}
for title, content in [doc.split(':') for doc in db]:
    words = word_tokenize(content)
    filtered_words = [stemmer.stem(w) for w in words if not w.lower() in stopWords and w != '.']
    processed_docs[title] = filtered_words

all_terms = set()
for terms in processed_docs.values():
    all_terms.update(terms)

documentVectorMatrix = []
for doc_title, terms in processed_docs.items():
    doc_vector = [binary_tf_idf(term, ' '.join(terms)) for term in all_terms]
    documentVectorMatrix.append(doc_vector)

df = pd.DataFrame(documentVectorMatrix)
df['title'] = dbDocTitles
df.set_index('title', inplace=True)
df.columns = list(all_terms)


initial_query = ['information', 'retrieval']
relevant_docs = ['information retrieval']
irrelevant_docs = ['search']


adjusted_query = rocchio_algorithm(initial_query, relevant_docs, irrelevant_docs)

print("Original Query:", initial_query)
print("Adjusted Query:", adjusted_query)
