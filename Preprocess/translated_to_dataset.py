import pickle
import spacy 
import numpy as np
from data_preprocess import add_word, is_intrans
from utils import remove_dups
from itertools import combinations

nlp = spacy.load('en_core_web_lg')

with open('SVO.p', 'rb') as f:
    svo = pickle.load(f)
    
with open('translated.p', 'rb') as t:
    paraphrases = pickle.load(t)
    
verb_dict = svo[0]
obj_dict = svo[1]
relations = svo[2] # ?
pairs = svo[3]
remove_dups(pairs)
print(len(verb_dict))
print(len(obj_dict))
print(len(pairs))
print('----------------')

for paraphrase_set in paraphrases:
    # verb-object index tuples
    vos = []
    
    # In first two sentences, we already have SVO mappings
    for sentence in paraphrase_set[:2]:
        doc = nlp(sentence)
        verb, obj = doc[0], doc[1]
        vi = add_word(verb_dict, [verb])
        oi = add_word(obj_dict, [obj])
        vos.append((vi,oi))
        
    # In the rest of the sentences, new dict entries might be necesssary    
    for sentence in paraphrase_set[2:]:
        doc = nlp(sentence)
        if len(doc) == 0: continue
        it =  is_intrans(doc)
        if it: 
            print(doc)
            verb = it[0]
            obj = it[1]
            vi = add_word(verb_dict, [verb])
            oi = add_word(obj_dict, [obj])
            vos.append((vi,oi))
            
    # verb-object tuples define a new list of pairs
    combs = combinations(vos, 2)
    for comb in combs:
        listedcomb = [comb[0][0], comb[0][1], comb[1][0], comb[1][1]]
        invertedlistcomb = [listedcomb[2], listedcomb[3], listedcomb[0], listedcomb[1]]

        if listedcomb not in pairs and invertedlistcomb not in pairs:
            pairs.append(listedcomb)

remove_dups(pairs)     
       
with open('tSVO.p', 'wb') as f:
    pickle.dump([verb_dict, obj_dict, pairs], f)