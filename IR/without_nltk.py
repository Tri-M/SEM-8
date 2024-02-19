import math
import pandas as pd

# Given database and documents to add
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

# Extract document titles from the database
dbDocTitles = [doc.split(':')[0] for doc in db]

# Append new documents to the database if their titles are not already present
for doc in docsToAdd:
    if doc.split(':')[0] not in dbDocTitles:
        db.append(doc)
        dbDocTitles.append(doc.split(':')[0])

# Function for tokenization
def tokenize(text):
    return text.lower().split()

# Function for stop word removal
stopWords = set(['a', 'an', 'the', 'is', 'of', 'in', 'on', 'and', 'to', 'by', 'be', 'as'])
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stopWords]

# Function for stemming (using Porter Stemmer algorithm)
def stem(word):
    if len(word) > 1 and word.endswith('ss'):
        return word[:-1]
    elif word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('s'):
        return word[:-1]
    return word

# Extract terms from documents, apply tokenization, stop word removal, and stemming
allTerms = set()
stemmedDB = {}
for doc in db:
    title, content = doc.split(':')
    tokens = tokenize(content)
    tokens = remove_stopwords(tokens)
    stemmed_words = [stem(token) for token in tokens]
    stemmedDB[title] = stemmed_words
    allTerms.update(stemmed_words)

# Function for term frequency calculation for Boolean Model
def tf_boolean(t, d):
    return int(t in d)

# Function for term frequency calculation for Binary Independence Model (BIM)
def tf_bim(t, d):
    return 1 if t in d else 0

# Weight calculation for Boolean Model
def boolean_model_weight(t, d):
    return tf_boolean(t, d)

# Weight calculation for Binary Independence Model (BIM)
def bim_weight(t, d):
    tf_value = tf_bim(t, d)
    idf_value = math.log((len(db) + 1) / (1 + sum(1 for doc in db if t in doc)))
    return tf_value * idf_value

# Select the weight calculation function based on the model
weight_function = bim_weight  # Change to boolean_model_weight for Boolean Model

# Calculate document vectors
documentVectorMatrix = [[weight_function(term, doc) for term in allTerms] for doc in stemmedDB.values()]

# Create DataFrame for document vectors
df = pd.DataFrame(documentVectorMatrix)
df['title'] = dbDocTitles
df.set_index('title', inplace=True)
df.columns = sorted(allTerms)
print(df)
