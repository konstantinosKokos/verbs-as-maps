from functools import reduce
from google.oauth2 import service_account
from google.cloud import translate

def init_credentials(filepath):
    return service_account.Credentials.from_service_account_file(filepath) # Point this to the json key file

def init_translator(credentials):
    return translate.Client(credentials=credentials) # Returns a translate.Client class object

def translate(translator, string, source, target):
    return translator.translate(values=string, source_language=source, target_language=target)
    
def chaintranslate(translator, languages, string):
    # foldl the original text through the translator
    return reduce(lambda string, source, target:
                         translate(translator, string, source, target),
                  ['en']+languages, languages[::-1], string)
    
def backtranslate(translator, languages, string):
    # Return to english
    return translate(translator, chaintranslate(translator, languages, string),
                    languages[::-1], 'en')
                    

