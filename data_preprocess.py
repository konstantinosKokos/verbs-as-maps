import spacy as sc 
import numpy as np
import pickle
from tqdm import tqdm

def read_pbdb(filename='ppdb-2.0-xxl-phrasal')
    s1s = []
    s2s = []
    with open(filename, 'r') as f:
        for line in f:
            s1, s2 = line.split('|||')[1:3]
            if len(s1.split())>1 and len(s2.split())>1:
                s1s.append(s1)
                s2s.append(s2)
    return s1s, s2s

def is_intrans(d, verbose=False):
    root = list(filter(lambda x: x.head==x, [t for t in d]))[0]
    if root.pos_ == 'VERB': 
        verb = root
        if verbose: print(verb)
        children = list(filter(lambda x: x.root.dep_ == 'dobj' and x.root.head == root, d.noun_chunks))
        if len(children) == 1:
            return verb, children[0].root
    return False

def add_word(voc, words):
    l = len(voc)
    for word in words:
        if word.lemma_ not in voc.values():
            l += 1
            voc[l] = word.lemma_
            
def add_relation(relations, pairs, verbs, objects, v1, o1, v2, o2):
    v1i = [k for k in verbs.keys() if verbs[k]==v1.lemma_][0]
    o1i = [k for k in objects.keys() if objects[k]==o1.lemma_][0]
    v2i = [k for k in verbs.keys() if verbs[k]==v2.lemma_][0]
    o2i = [k for k in objects.keys() if objects[k]==o2.lemma_][0]
    if v1i not in relations.keys(): relations[v1i] = [o1i]
    elif o1i not in relations[v1i]: relations[v1i].append(o1i)
    if v2i not in relations.keys(): relations[v2i] = [o2i]
    elif o2i not in relations[v2i]: relations[v2i].append(o2i)
    if [v1i, o1i, v2i, o2i] not in pairs and (v1i != v2i or o1i != o2i):
        pairs.append([v1i, o1i, v2i, o2i])

def parse_phrases(s1s, s2s, nlp, dumpfile='SVO.p')
    s1batch = []
    s2batch = []

    for i, s1 in tqdm(enumerate(s1s)):
        if i <= c: continue
        s2 = s2s[i]
        if len(s1)>1 and len(s2)>1:
            s1batch.append(s1)
            s2batch.append(s2)
        if len(s1batch) == 500:
            docs1 = nlp.pipe(s1batch, batch_size=100, n_threads=4)
            docs2 = nlp.pipe(s2batch, batch_size=100, n_threads=4)
            for (d1,d2) in zip(docs1,docs2):
                if is_intrans(d1) and is_intrans(d2):
                    v1, o1 = is_intrans(d1)
                    v2, o2 = is_intrans(d2)
                    add_word(verbs, [v1, v2])
                    add_word(objects, [o1, o2])
                    add_relation(relations, pairs, verbs, objects, v1, o1, v2, o2)
            s1batch = []
            s2batch = []
        if i%500000 == 0 and i!= 0:
            print('Saving at {}'.format(i))
            with open(dumpfile, 'wb') as f:
                pickle.dump((verbs,objects,relations,pairs,i), f)