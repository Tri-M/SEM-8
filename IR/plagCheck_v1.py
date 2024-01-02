from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')

ps = PorterStemmer()

db = [
    ("information requirement", "query considers the user feedback as information requirement to search."),
    ("information retrieval", "query depends on the model of information retrieval used."),
    ("prediction problem", "Many problems in information retrieval can be viewed as prediction problems."),
    ("search", "A search engine is one of applications of information retrieval models.")
]

def stem_title(title):
    words = nltk.word_tokenize(title)
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

def cosineSimilarity(u, v):
    vectorizer = CountVectorizer().fit_transform([u, v])
    similarity_matrix = cosine_similarity(vectorizer)
    return similarity_matrix[0, 1]

def plagCheck(doc, db, threshold=0.85):
    title, text = doc.split(":")
    stemmed_title = stem_title(title)

    duplicate = any(cosineSimilarity(stemmed_title, stem_title(oldTitle)) > threshold for oldTitle, _ in db)
    
    if duplicate:
        print("Duplicate Document")
    else:
        db.append((title, text))
        print("Added to database")

flag = True
while flag:
    print("Enter 1 for input text and 0 to exit: ")
    inp = int(input())
    
    if inp == 1:
        newDoc = input("Enter document in the form of title:text - ")
        plagCheck(newDoc, db)
        print(db)
    else:
        flag = False
