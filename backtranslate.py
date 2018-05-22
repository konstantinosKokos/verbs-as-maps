from functools import reduce
from google.oauth2 import service_account
from google.cloud import translate
import numpy as np
import argparse
from tqdm import tqdm
import pickle
from data_preprocess import is_intrans
from utils import remove_dups

def init_credentials(filepath):
    return service_account.Credentials.from_service_account_file(filepath) # Point this to the json key file

def init_translator(credentials):
    return translate.Client(credentials=credentials) # Returns a translate.Client class object

def translate_once(translator, string, source, target):
    return translator.translate(values=string, source_language=source, target_language=target)['translatedText']
  
def chaintranslate(translator, languages, string):
    # ease-of-use conversion to list
    languages = list(languages)
    # foldl the original text through the translator
    return reduce(lambda string, pair:
                         translate_once(translator, string, pair[0], pair[1]), # The reduction operator returns strings
                  zip(['en']+languages, languages), # The iterable is a list of language tuples
                  string) # The seed is the input 

def backtranslate(translator, languages, string):
    # Return to english
    return translate_once(translator, chaintranslate(translator, languages, string),
                    languages[-1], 'en')
  
def randomchain(languages, maxchain):
    return tuple(np.random.permutation(languages)[:(1+np.random.randint(maxchain))].tolist())

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Back-translation utilities for dataset enhancement.')
    parser.add_argument('-c', type=str, help='file containing GCA credentials', nargs='?')
    parser.add_argument('-f', type=str, help='file containing existing data', nargs='?')
    parser.add_argument('-p', type=str, help='file containing existing paraphrases', nargs='?')
    
    all_languages = ['sw', 'es', 'ru', 'no', 'it', 'fr', 'nl', 'el', 'sk', 'tr', 'de', 'ar']    
    
    if args.p is not None:
        with open(args.p, 'rb') as f:
            paraphrases = pickle.load(f)
        else:
            paraphrases = []
    
    with open(args.f, 'rb') as f:
        svo = pickle.load(f)
    
    pairs = svo[3]
    remove_dups(pairs)
    
    for pair_index, pair in tqdm(enumerate(pairs)):
        if pair_index < len(paraphrases): continue
            v0 = pair[0]
            o0 = pair[1]
            v1 = pair[2]
            o1 = pair[3]

    # Text pairs
    text0 = svo[0][v0] + ' ' + svo[1][o0]
    text1 = svo[0][v1] + ' ' + svo[1][o1]
    
    # Random chains
    forward_chains = list(set([randomchain(all_languages, 5) for i in range(6)]))
    
    # Obtain new paraphrases
    local_paraphrases =  [text0, text1] # Add the original texts
    for fc in forward_chains:
        for text in [text0, text1]:
            translation = backtranslate(translator, fc, text)
            if translation not in local_paraphrases: local_paraphrases.append(translation)
    paraphrases.append(local_paraphrases)
    
    with open('translated.p', 'wb') as f:
        pickle.dump(paraphrases, f)