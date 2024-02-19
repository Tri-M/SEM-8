import pandas as pd

from itertools import combinations

minSup=0.5
conf=50

def getTrans(df):
    cols=df.columns
    trans=[]
    for _,row in df.iterrows():
        transaction=set()
        for col in cols:
            if row[col]=="Yes":
                transaction.add(col)
        trans.append(transaction)
    return trans

def getSuppCount(transactions,itemSets):
    count={}
    for itemSet in itemSets:
        itemSet=frozenset(itemSet)
        