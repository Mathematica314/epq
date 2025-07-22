import pickle
import random
from collections import defaultdict
import math

composers = ["bach", "beethoven", "chopin", "haydn", "joplin", "mozart", "scarlatti"]

with open("fets.pkl","rb") as file:
    dat = pickle.load(file)

def norm(mean,var,x):
    return math.exp(-0.5*(mean-x)**2/var)/math.sqrt(2*math.pi*var)

def prob(x,c,stats):
    st = stats[c]
    return st[0]*math.prod([norm(r[0],r[1],x_i) for r,x_i in zip(st[1:],x)])

def bayes(datapoint,stats):
    return max([(c,prob(datapoint,c,stats)) for c in composers],key=lambda x:x[1])[0]

def test():
    random.shuffle(dat)

    test = dat[:150]
    train = dat[150:]

    tr_comps = defaultdict(list)

    for x in train:
        tr_comps[x[0]].append(x[1])

    tr_comps = {k: list(zip(*v[::-1])) for k, v in tr_comps.items()}
    stats = {k: [len(v[0]) / sum([len(x[0]) for x in tr_comps.values()])] + [
        [sum(a) / len(a), (sum([x ** 2 for x in a]) / len(a) - (sum(a) / len(a)) ** 2)] for a in v] for k, v in
             tr_comps.items()}
    correct = 0
    for d in test:
        if bayes(d[1],stats) == d[0]:
            correct += 1
    return correct/len(test)

def multitest(n):
    total = sum([test() for _ in range(n)])
    return total/n

print(f"{round(multitest(100)*100,1)}%")