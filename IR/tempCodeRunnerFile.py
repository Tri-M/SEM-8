    precision = len(intersect) / len(set(predicted))
    recall = len(intersect) / len(set(actual))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1

cosSim_metrics = {q: metrics(cosSimRelDocs[q], givenRelDocs[q]) for q in givenRelDocs}
BIM_metrics = {q: metrics(BIMRelDocs[q], givenRelDocs[q]) for q in givenRelDocs}

metricDF = pd.concat([pd.DataFrame(m).T.assign(Algorithm=a) for m, a in [(cosSim_metrics, 'Cosine Similarity'), (BIM_metrics, 'BIM')]])
metricDF.columns = ['Precision', 'Recall', 'F1-Measure', 'Algorithm']
print(metricDF)