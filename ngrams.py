import os
import pickle
from collections import defaultdict
import random
import math

def getngrams(n,data):
    out = []
    for i in range(len(data)-n+1):
        out.append(tuple(data[i:i+n]))
    return out

def similarity(x,profile):
    total = 0
    for gram in x:
        if gram in list(profile[1].keys()):
            total += 4 - ((2*x[gram]-profile[1][gram])/(x[gram]+profile[1][gram]))**2
    return total

def p_differences(data):
    a = data[0]
    out = []
    for d in data[1:]:
        out.append(d-a)
        a = d
    return out

def r_differences(data):
    a = data[0]
    out = []
    for d in data[1:]:
        out.append(round(math.log2(d/a)*5)/5)
    return out

def get_profiles(data, n,seed=0):
    data = [(k, getngrams(n, v)) for k, v in data]
    random.seed(seed)
    random.shuffle(data)

    test = data[:150]
    train = data[150:]

    profiles = defaultdict(lambda: defaultdict(int))

    for comp in train:
        for g in comp[1]:
            profiles[comp[0]][g] += 1

    t_profiles = []

    for datapoint in test:
        d = defaultdict(int)
        for gr in datapoint[1]:
            d[gr] +=1
        d = {k:v for k,v in zip(d.keys(),d.values())}
        t_profiles.append((datapoint[0],d))

    profiles = [(k,{ka:va for ka,va in zip(v.keys(),v.values())}) for k,v in zip(profiles.keys(),profiles.values())]
    return profiles,t_profiles

def mono_ngram_analysis(data,n):
    profiles,t_profiles = get_profiles(data, n)
    total = 0
    for datapoint in t_profiles:
        if max(profiles,key = lambda x: similarity(datapoint[1],x))[0] == datapoint[0]:
            total += 1
    return total/len(t_profiles)

def dual_ngram_analysis(data, n):
    data = [(a[0],[x for y in list(zip(a[1],b[1])) for x in y]) for a,b in zip(data[0],data[1])]
    return data

def multi_analysis(p_difs,r_difs,n):
    combined = dual_ngram_analysis((p_difs,r_difs),n)
    pprof,t_pprof = get_profiles(p_difs,n)
    rprof,t_rprof = get_profiles(r_difs,n)
    cprof,t_cprof = get_profiles(combined,n*2)
    success = 0
    for tp,tr,tc in zip(t_pprof,t_rprof,t_cprof):
        scores = {}
        for p,r,c in zip(pprof,rprof,cprof):
            scores[p[0]] = similarity(tp[1],p)+similarity(tr[1],r)+similarity(tc[1],c)
        success += max(scores.keys(),key=lambda x:scores[x]) == tp[0]
    return success/len(t_pprof)

def multiseed(data,n,tries,analyser=mono_ngram_analysis):
    total = 0
    for _ in range(tries):
        total += analyser(data, n)
    return total/tries

p_data = []

p_unidir = os.fsencode("data/p_unigrams")

for composer in os.listdir(p_unidir):
    for file in os.listdir(os.path.join(p_unidir, composer)):
        with open(os.path.join(p_unidir, composer, file).decode(), "rb") as f:
            p_data.append((composer.decode(),pickle.load(f)))

r_unidir = os.fsencode("data/r_unigrams")

r_data = []

for composer in os.listdir(r_unidir):
    for file in os.listdir(os.path.join(r_unidir, composer)):
        with open(os.path.join(r_unidir, composer, file).decode(), "rb") as f:
            r_data.append((composer.decode(),pickle.load(f)))

difs = [(d[0],p_differences(d[1])) for d in p_data]
rdifs = [(d[0],r_differences(d[1])) for d in r_data]

for n in range(8,9):
    print(f"n={n}")
    print(mono_ngram_analysis(p_data,n))
    print(mono_ngram_analysis(r_data,n))
    print(mono_ngram_analysis(difs,n))
    print(mono_ngram_analysis(rdifs,n))
    print(multiseed((difs,rdifs), n, 1, analyser=lambda d,n: mono_ngram_analysis(dual_ngram_analysis(d,n),n*2)))
    print(multi_analysis(difs,rdifs,n))
