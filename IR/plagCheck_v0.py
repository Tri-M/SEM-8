db=[("title1", "text1"),("title2","text2")]

def binaryDistance(u,v):
    uSet=set(u.lower().split())
    vSet=set(v.lower().split())
    interSec=len(uSet.intersection(vSet))
    union=len(uSet.union(vSet))
    return 1-(interSec/union)

def plagCheck(doc,db):
    title,text=doc.split(":")
    dup=any(binaryDistance(title,oldTitle)==0 for oldTitle, _ in db)
    if dup:
        print("Duplicate Document")
    else:
        db.append((title,text))
        print("Added to database")
flag=True
while flag:
    print("Enter 1 for input text and 0 to exit: ")
    inp=int(input())
    if inp==1:
        newDoc=input("Enter document in form of title:text -  ")
        newText=plagCheck(newDoc, db)
        print(db)
    else:
        flag=False