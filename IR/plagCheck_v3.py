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

### A) Verify if the titles are exactly same (Apply BinaryDistance(u,v), which gives the
# binary distance between vectors u and v, equal to 0 if they are identical and 1
# otherwise.). If same, label the document as duplicate and discard it else proceed to
# second part of the Checker.

dbDocTitles = [doc.split(':')[0] for doc in db]

for doc in docsToAdd:
   if doc.split(':')[0] not in dbDocTitles:
    db.append(doc)
    dbDocTitles.append(doc.split(':')[0])
   
import math
# Term Frequency
def tf(t, d):
    return d.split().count(t)/len(d.split())

# Document Frequency
def docfreq(t):
    termCount = 0
    for doc in db:
        termCount += doc.split(':')[1].split().count(t)
    return termCount

# Modified Inverse Document Frequency
def mod_idf(t):
    # Number of documents containing term t
    N = 0
    for doc in db:
        if t in doc.split(':')[1].split():
            N += 1
    return math.log((N + 1) / (0.5 + docfreq(t)))

# Given weight formula
# tf-idf(t, d) = tf(t, d) * mod_idf(t)
def tf_mod_idf(t, d):
    return tf(t, d) * mod_idf(t)

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stopWords = set(stopwords.words('english'))

docsMap = {doc.split(':')[0]:doc.split(':')[1] for doc in db}


# Stop word removal
stopWordRemovedResult = {}
for title, content in docsMap.items():
    words = word_tokenize(content)
    filtered_words = [w for w in words if not w.lower() in stopWords and w != '.']
    stopWordRemovedResult[title] = filtered_words
   
# Stemming
porter = PorterStemmer()
stemmedListMap = {}
for title, wordList in stopWordRemovedResult.items():
    stemmedWords = []
    for word in wordList:
        stemmedWord = porter.stem(word)
        stemmedWords.append(stemmedWord)
    stemmedListMap[title] = stemmedWords
stemmedListMap

# Joining
stemmedDB = {title: ' '.join(content) for title, content in stemmedListMap.items()}
allTerms = sorted(list(set([term for doc in stemmedListMap.values() for term in doc])))

# Weight calc
documentVectorMatrix = [[tf_mod_idf(term, doc) for term in allTerms] for doc in stemmedDB.values()]
documentVectorMatrix

df = pd.DataFrame(documentVectorMatrix)
df['title'] = dbDocTitles
df.set_index('title', inplace=True)
df.columns = allTerms
df

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stopWords = set(stopwords.words('english'))

docsMap = {doc.split(':')[0]:doc.split(':')[1] for doc in db}


# Stop word removal
stopWordRemovedResult = {}
for title, content in docsMap.items():
    words = word_tokenize(content)
    filtered_words = [w for w in words if not w.lower() in stopWords and w != '.']
    stopWordRemovedResult[title] = filtered_words
   
# Stemming
porter = PorterStemmer()
stemmedListMap = {}
for title, wordList in stopWordRemovedResult.items():
    stemmedWords = []
    for word in wordList:
        stemmedWord = porter.stem(word)
        stemmedWords.append(stemmedWord)
    stemmedListMap[title] = stemmedWords
stemmedListMap

# Joining
stemmedDB = {title: ' '.join(content) for title, content in stemmedListMap.items()}
allTerms = sorted(list(set([term for doc in stemmedListMap.values() for term in doc])))

# Weight calc
documentVectorMatrix = [[tf_mod_idf(term, doc) for term in allTerms] for doc in stemmedDB.values()]
documentVectorMatrix

df = pd.DataFrame(documentVectorMatrix)
df['title'] = dbDocTitles
df.set_index('title', inplace=True)
df.columns = allTerms
print(df)