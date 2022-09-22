import math

from sklearn import datasets as ds
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    lE = LabelEncoder()
    df = pd.read_csv("Naive Bayes.csv")
    df[["Outlook", "Temp", "Hum", "Wind",  "Play"]] = df[["Outlook", "Temp", "Hum", "Wind",  "Play"]].apply(lE.fit_transform)
    #print(df.head(5))
    X = df.drop('Play', axis=1)
    X = np.array(X)
    y = df.drop(labels=["Outlook", "Temp", "Hum", "Wind"], axis=1)
    y = np.array(y)
    #print(y)
    u = np.unique(X[:, 0])
    #print(X[:, 3])
    u = np.unique(y)
    sp = X.shape
    #print(sp[1])
    #print(sp[1])
    lst = [] #np.array([np.array([np.array([0 for b in np.unique(X[:, i])])for i in range(sp[1])]) for x in u])
    '''c = X[:, 0]
    index = (y == 0).flatten()

    C = c[index]
    print(index)
    print(C)
'''
    p = []
    for i in u:

        index = (y == i).flatten()
        p.append(len(y[index])/len(y))
        mmList = []
        for f in range(sp[1]):
            colmn = (X[:, f])
            #for colmn in (X[:, f]):
            #print(colmn)
            mainClm = colmn[index]
            lan = len(mainClm)
            mList = []
            for j in np.unique(colmn):
                lt = mainClm[mainClm == j]
                #print(lt)
                lng = len(lt)
                #prior = math.log10(lng/lan)
                prior = lng/lan
                mList.append(prior)
            mmList.append(mList)
        lst.append(mmList)
    #print(lst)

    inputs = [2, 0, 0, 0]

    cmList = []

    for indx, i in enumerate(lst):
        sum = 0
        for inx, f in enumerate(i):
            sum += math.log10(f[inputs[inx]])
        sum += p[indx]
        cmList.append(sum)
    #print(cmList)
    maxValu = max(cmList)
    indd = cmList.index(maxValu)
    if(indd == 0):
        print("No")
    else:
        print("Yes")
