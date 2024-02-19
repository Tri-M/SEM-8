import numpy as np
import pandas as pd
import re
from nltk.stem import PorterStemmer
import math
import collections

with open('TIME.ALL', 'r') as f:
    text = f.read()
result = re.findall(r"\*TEXT\s+(\d{3})\s+(\d{2}/\d{2}(/|\s)\d{2})\s+PAGE\s+(\d{3})\n\n(.+?)(?=\*TEXT|$)", text, re.DOTALL)
docDB = {match[0]: {'id': match[0], 'date': match[1], 'page': int(match[3]), 'text': match[4]} for match in result}
docOnlyDB = {match[0]: match[4] for match in result}
docDF = pd.DataFrame(docDB).T

# Queries
with open('TIME.QUE', 'r') as f:
    text = f.read()
queryDB = re.findall(r'FIND\s+\d+\s+(.+?)(?=\n\n\*FIND\s+\d+\s+|$)', text, re.DOTALL)

# Stopwords
with open('TIME.STP','r') as f:
    text = f.read()
swDB = re.findall(r"^[A-Z]+$", text, re.MULTILINE)
swDB = set([word.lower() for word in swDB])

# Relevant docs
with open('TIME.REL', 'r') as f:
    text = f.read()
    lines = text.split("\n")
rdDB = {}
for line in lines:
    numbers = re.findall(r"\d+", line)
    if numbers:
        key = numbers[0]
        values = [int(n) for n in numbers[1:]]
        rdDB[key] = values
# Tokenize a document
def tokenizeDocument(documentText):
    return documentText.split()

# Normalize and Stop the token stream of a document
def normalizeAndStopTokenStream(tokenStream):
    return [token.lower() for token in tokenStream if token.lower().isalnum() and token.lower() not in swDB]

# Stem and the normalized token stream of a document
def stemNormalizedTokenStream(normalizedTokenStream):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(token) for token in normalizedTokenStream])

def processDocument(documentText):
    return stemNormalizedTokenStream(normalizeAndStopTokenStream(tokenizeDocument(documentText)))
processedDocDB = {docID: processDocument(text) for docID, text in docOnlyDB.items()}

allTerms = sorted(list(set([token for doc in processedDocDB.values() for token in doc.split()])))

def df(term, documentDB):
    return len([1 for document in documentDB if term in document])

# Inverse document frequency or informativeness of a term
def idf(term, documentDB):
    return math.log((len(documentDB) + 1) / (df(term, documentDB) + 0.5))

# Precalculating and making a cache
idfMap = {term:idf(term, processedDocDB.values()) for term in allTerms}
def getIDF(term):
    return idfMap[term]

# Calculate weights for a document
def calculateDocumentWeight(document):
    counter = collections.Counter(document.split())
    return [counter.get(term, 0) * getIDF(term) for term in allTerms]
tdMatrixDF = pd.DataFrame([calculateDocumentWeight(processedDoc) for processedDoc in processedDocDB.values()], columns=allTerms, index=processedDocDB.keys())
processedQueryDB = [processDocument(query) for query in queryDB]

def cosineSimiliary(queryWeights, tdMatrix):
    return np.dot(tdMatrix, queryWeights) / (np.linalg.norm(queryWeights) * np.linalg.norm(tdMatrix, axis=1))

def findTopNRelevantDocsWithCosineSimilarity(query, tdMatrixDF, N):
    # Calculate the cosine similarity
    cosineSimilarities = cosineSimiliary(calculateDocumentWeight(query), tdMatrixDF.values)

    # Sort in descending order of cosine similarity
    df = pd.DataFrame({'docID': tdMatrixDF.index, 'cosineSimilarity': cosineSimilarities})
    sorted_df = df.sort_values(by='cosineSimilarity', ascending=False)

    # Return the top 10 relevant documents from the sorted dataframe
    return sorted_df['docID'].values[:N].tolist()
cosineSimiliaryResults = {str(idx + 1):findTopNRelevantDocsWithCosineSimilarity(processedQuery, tdMatrixDF, 10) for idx, processedQuery in enumerate(processedQueryDB)}

def trk(term, relevantDocsDB, nonRelevantDocsDB):
    rk = df(term, relevantDocsDB.values())
    nrk = df(term, nonRelevantDocsDB.values())
    Opk = (rk+0.5) / (len(relevantDocsDB.values())-rk+0.5)
    Oqk = (nrk+0.5) / (len(nonRelevantDocsDB.values())-nrk+0.5)
    return np.log10(Opk/Oqk)

# Phase 1 of BIM assumes all the documents to be non-relevant when computing the term score (t<sub>rk</sub>)
trkMap = {term:trk(term, {}, processedDocDB) for term in allTerms}

def getWeightVector(trk, document):
    tokenStream = set(document.split())
    return [trk[term] if term in tokenStream else 0 for term in allTerms]

def generateTDMDataFrame(trk, processedDocDB):
    return pd.DataFrame([getWeightVector(trk, processedDoc) for processedDoc in processedDocDB.values()], columns=allTerms, index=processedDocDB.keys())
def simpleSimiliaryForBIM(query, tdMatrix):
    queryWeights = getWeightVector(trkMap, query)
    return np.dot(tdMatrix, queryWeights)

def findTopNRelevantDocsWithBIM(query, tdMatrix, N):
    similarities = simpleSimiliaryForBIM(query, tdMatrix.values)
    
    df = pd.DataFrame({'docID': tdMatrix.index, 'similarity': similarities})
    sorted_df = df.sort_values(by='similarity', ascending=False)
    return sorted_df['docID'].values[:N].tolist()
def BIMPhase1(trk, queryDB, documentDB, N):
    tdMatrix = generateTDMDataFrame(trk, documentDB)
    return {str(queryID + 1):findTopNRelevantDocsWithBIM(query, tdMatrix, N) for queryID, query in enumerate(queryDB)}
BIMPhase1Top10 = BIMPhase1(trkMap, processedQueryDB, processedDocDB, 10)

def recomputeTrk(relDocs, documentDB):
    relevantDocs, nonRelevantDocs = {}, {}
    relDocSet = set(relDocs)
    for docID, text in documentDB.items():
        if docID in relDocSet:
            relevantDocs[docID] = text
        else:
            nonRelevantDocs[docID] = text
    return {term:trk(term, relevantDocs, nonRelevantDocs) for term in allTerms}
def BIMPhase2(BIMPhase1Result, queryDB, documentDB):
    BIMPhase2TopN = {}
    for queryID, query in enumerate(queryDB):
        queryRelDocsFromPhase1 = BIMPhase1Result[str(queryID+1)]
        trkAdjusted = recomputeTrk(queryRelDocsFromPhase1, documentDB)
        tdMatrixAdjusted = generateTDMDataFrame(trkAdjusted, documentDB)
        BIMPhase2TopN[str(queryID+1)] = findTopNRelevantDocsWithBIM(query, tdMatrixAdjusted, 10)
    return BIMPhase2TopN
BIMPhase2Top10 = BIMPhase2(BIMPhase1Top10, processedQueryDB[:5], processedDocDB)

givenRelDocs = dict(list(rdDB.items())[:5])

cosSimRelDocs = dict(list(cosineSimiliaryResults.items())[:5])
BIMRelDocs = BIMPhase2Top10
def metrics(predicted, actual):
    intersect = set(predicted) & set(actual)
    precision = len(intersect) / len(set(predicted))
    recall = len(intersect) / len(set(actual))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1

cosSim_metrics = {q: metrics(cosSimRelDocs[q], givenRelDocs[q]) for q in givenRelDocs}
BIM_metrics = {q: metrics(BIMRelDocs[q], givenRelDocs[q]) for q in givenRelDocs}

metricDF = pd.concat([pd.DataFrame(m).T.assign(Algorithm=a) for m, a in [(cosSim_metrics, 'Cosine Similarity'), (BIM_metrics, 'BIM')]])
metricDF.columns = ['Precision', 'Recall', 'F1-Measure', 'Algorithm']