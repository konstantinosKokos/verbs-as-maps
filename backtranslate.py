from functools import reduce
from google.oauth2 import service_account
from google.cloud import translate

def init_credentials(filepath):
    return service_account.Credentials.from_service_account_file(filepath) # Point this to the json key file

def init_translator(credentials):
    return translate.Client(credentials=credentials) # Returns a translate.Client class object

def translate(translator, string, source, target):
    return translator.translate(values=string, source_language=source, target_language=target)['translatedText']
  
def chaintranslate(translator, languages, string):
    # foldl the original text through the translator
    return reduce(lambda string, pair:
                         translate(translator, string, pair[0], pair[1]), # The reduction operator returns strings
                  zip(['en']+languages, languages), # The iterable is a list of language tuples
                  string) # The seed is the input 

def backtranslate(translator, languages, string):
    # Return to english
    return translate(translator, chaintranslate(translator, languages, string),
                    languages[::-1], 'en')tring
  

