import pickle
import random
from collections import defaultdict
import math

def normal(mean,variance,x):
    return math.exp(-0.5*(x-mean)**2/variance)/(math.sqrt(variance*2*math.pi))

def classify(p,vals,comps,train):
    dens = []
    for c in vals.keys():
        p_dens = len(comps[c][0])/len(train)
        for chr,prms in zip(p[1],vals[c]):
            p_dens *= normal(prms[0],prms[1],chr)
        dens.append((c,p_dens))
    return max(dens,key=lambda x:x[1])[0]

def test(data):
    random.shuffle(data)
    test = data[:150]
    train = data[150:]

    comps = defaultdict(list)

    for datapoint in train:
        comps[datapoint[0]].append(datapoint[1])

    comps = {k: list(zip(*v[::-1])) for k, v in comps.items()}

    normal_values = {k: [(sum(a) / len(a), sum([x ** 2 for x in a]) / len(a) - (sum(a) / len(a)) ** 2) for a in v] for
                     k, v in comps.items()}

    total = 0
    for p in test:
        if classify(p,normal_values,comps,train) == p[0]:
            total += 1

    return (total / 150)

def multiseed(data,n):
    total = 0
    for i in range(n):
        total += test(data)
    return total/n

with open("fets.pkl","rb") as f:
    data = pickle.load(f)

print(multiseed(data,100))