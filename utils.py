import numpy as np
from keras.utils import to_categorical
import spacy
from matplotlib import pyplot as plt

nlp = spacy.load('en_vectors_web_lg')

def flatten(relations):
    # flatten relations
    return [item for sublist in [[(i,j) for j in relations[i]] for i in relations] for item in sublist]

def remove_dups(pairs):
    # hold unique pairs only
    for i, p0 in enumerate(pairs):
        for j, p1 in enumerate(pairs):
            if p0[2] == p1[0] and p0[3] == p1[1] and p0[0] == p1[2] and p0[1] == p1[3]:
                del(pairs[j])

def data_generator(verbs, objects, pairs, batch_size = 64, random_chance = 0.5, return_signatures=False):
    """
    Iterates over vo paraphrase pairs, yielding their corresponding label
    """
    
    current_index = 0
    num_verbs = len(verbs)+1
    
    v0, o0, v0sig, v1, o1, v1sig, t = [], [], [], [], [], [], []
    
    while True:
        if np.random.random() > random_chance:
            current_sample = pairs[current_index]

            if current_index == len(pairs)-1: current_index = 0
            else: current_index += 1
                
            t.append(1)
        else:
            current_sample = [np.random.randint(len(verbs))+1, np.random.randint(len(objects))+1,
                             np.random.randint(len(verbs))+1, np.random.randint(len(objects))+1]
            if ([current_sample[2], current_sample[3], current_sample[0], current_sample[1]] in pairs 
                or current_sample in pairs): # Make sure that the random sample isn't actually a paraphrase
                t.append(1)
            else:
                t.append(0)
        
        v0.append(nlp(verbs[current_sample[0]]).vector)
        o0.append(nlp(objects[current_sample[1]]).vector)
        v0sig.append(to_categorical(current_sample[0], num_verbs))
        v1.append(nlp(verbs[current_sample[2]]).vector)
        o1.append(nlp(objects[current_sample[3]]).vector)
        v1sig.append(to_categorical(current_sample[2], num_verbs))
        
        if len(v0) == batch_size:
            v0, o0, v0sig, v1, o1, v1sig, t = (np.array(v0), np.array(o0), np.array(v0sig), 
                                               np.array(v1), np.array(o1), np.array(v1sig),
                                               np.array(t))
            if return_signatures: yield [v0, o0, v0sig, v1, o1, v1sig], t
            else: yield [v0, o0, v1, o1], t
            v0, o0, v0sig, v1, o1, v1sig, t = [], [], [], [], [], [], []
            
def histplot(history):
    for key in history:
        plt.plot((history[key]))
    plt.legend([key for key in history.keys()])
    plt.show()