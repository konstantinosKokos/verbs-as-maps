import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
import spacy

nlp = spacy.load('en_vectors_web_lg')

# Open the translated, duplicate-clear dataset
with open('tSVO_rd.p', 'rb') as f:
    verbs, objects, pairs, = pickle.load(f)
    
# Instantiate an empty dictionary linking each verb to objects it appears with
relations = {k: set() for k in verbs.keys()}

# Count the occurrences for each verb and object
verb_counts = {k: 0 for k in verbs.keys()}
object_counts = {k: 0 for k in objects.keys()}
for pair in pairs:
    verb0, verb1 = pair[0], pair[2]
    object0, object1 = pair[1], pair[3]
    relations[verb0] = relations[verb0]|{object0}
    relations[verb1] = relations[verb1]|{object1}
    verb_counts[verb0] = verb_counts[verb0]+1
    verb_counts[verb1] = verb_counts[verb1]+1
    object_counts[object0] = object_counts[object0]+1   
    
# Instantiate two empty lists of indices to remove
verbs_to_delete = []
objects_to_delete = []

# Ad-Hoc Deletion: Remove bad words through manual inspection
bad_verbs = ['s', 've', "'d", 'ca', 'wo', '\u200b\u200bup', "n'thave", 'ai', 'binyamin', 'wanna', 'mobutu']
for i, k in verbs.items():
    if k in bad_verbs: verbs_to_delete.append(i)
        
bad_objects = ['-PRON-']
for i, k in objects.items():
    if k in bad_objects: objects_to_delete.append(i)
        
#  Remove OOV instances (instances where the parser classified a tuple as VO, but no vector was assigned)
# Non-word objects
non_word_objects = []
for k, obj in enumerate(objects.values()):
    if np.sum(nlp(obj).vector) == 0: 
        non_word_objects.append(k)
objects_to_delete.extend(non_word_objects)
# Non-word verbs
non_word_verbs = []
for k, verb in enumerate(verbs.values()):
    if np.sum(nlp(verb).vector) == 0: 
        non_word_verbs.append(k)     
verbs_to_delete.extend(non_word_verbs)

# Mutual Information Count: Measure the universality of each verb
# Relation-values
relation_values = np.array([len(k) for k in relations.values()])
# For each verb, count how many verbs have more unique counts
percentiles = {k: len(relation_values[relation_values>len(relations[k])])/len(relation_values)
               for k in verbs.keys()}
# Delete both tails of the distribution
verbs_to_delete.extend([i for i,k in percentiles.items() if k<0.05 or k>0.95])

# Occurrence Count: Measure the rarity of each verb
# Count-values
count_values = np.array([k for k in verb_counts.values()])
# For each verb, count how many verbs have more unique counts
percentiles = {k: len(count_values[count_values>verb_counts[k]])/len(count_values)
               for k in verbs.keys()}
# Delete both tails of the distribution
verbs_to_delete.extend([i for i,k in percentiles.items() if k<0.05 or k>0.95])

# Scan the pairs for usage of verbs or objects that will be removed
pairs_to_delete = []
for i, pair in enumerate(pairs):
    verb0, verb1 = pair[0], pair[2]
    object0, object1 = pair[1], pair[3]
    if verb0 in verbs_to_delete or verb1 in verbs_to_delete:
        pairs_to_delete.append(i)
    elif object0 in objects_to_delete or object1 in objects_to_delete:
        pairs_to_delete.append(i)
        
# Remove bad pairs
filtered = [pairs[i] for i in range(len(pairs)) if i not in pairs_to_delete]
print(len(filtered))
assert(len(pairs) == len(filtered) + len(pairs_to_delete))

# Reindex verbs and objects
filtered_verbs = {k: v for k,v in verbs.items() if k not in verbs_to_delete}
assert(len(verbs) == len(set(verbs_to_delete))+len(filtered_verbs))
reindexed_verbs = {i+1: v for i, v in enumerate(filtered_verbs.values())}
verb_reindexing_map = {k: i+1 for i, k in enumerate(filtered_verbs.keys())}

filtered_objects = {k: o for k,o in objects.items() if k not in objects_to_delete}
assert(len(objects) == len(set(objects_to_delete)) + len(filtered_objects))
reindexed_objects = {i+1: o for i,o in enumerate(filtered_objects.values())}
object_reindexing_map = {k: i+1 for i, k in enumerate(filtered_objects.keys())}

# Reconstruct the dataset with the reduced sets
for pair in filtered:
    pair[0] = verb_reindexing_map[pair[0]]
    pair[2] = verb_reindexing_map[pair[2]]
    pair[1] = object_reindexing_map[pair[1]]
    pair[3] = object_reindexing_map[pair[3]]
    
# Store the sanitized dataset
with open('tSVO_sanitized.p', 'wb') as f:
    pickle.dump([reindexed_verbs, reindexed_objects, filtered], f)