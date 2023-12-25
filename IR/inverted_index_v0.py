f1="Hello world. Hello semester 8. good morning. Greetings."
f2="Hello. apple. morning. good night."

terms1=f1.lower().split()
terms2=f2.lower().split()

terms=list(set(terms1+terms2))

invIndex={}

for term in terms:
    files=[]
    if term in terms1:
        files.append("1")
    if term in terms2:
        files.append("2")
    invIndex[term]=files

for term, files in invIndex.items():
    print(term, "-->", ",".join(files))

