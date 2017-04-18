
# coding: utf-8

# In[ ]:

import pandas as pd;
import numpy as np;
import scipy as sp;
import re;
import pickle;
import os;
import subprocess

import sklearn;
from sklearn.utils import shuffle;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;


# In[ ]:

def notify_slack(text):
    subprocess.Popen('''curl -X POST --data-urlencode "payload={'channel' : '#random', 'username': 'webhookbot', 'text':'''+ '\'' + text + '\'' + '''}" https://hooks.slack.com/services/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''', shell=True)
 


# In[ ]:

try:
    cleaned_tfidf_counts = pickle.load(open("sparse_tfidf",'rb'))
    
    notify_slack("Converting to array")
    cleaned_tfidf_counts_array = cleaned_tfidf_counts.toarray()
    
    notify_slack("Pickling array into current directory")
    #pickle.dump(cleaned_tfidf_counts_array, open("/home/ubuntu/CliNER/data/saved/tfidf_vectors",'wb'))
    pickle.dump(cleaned_tfidf_counts_array, open("tfidf_vectors",'wb'))

    notify_slack("Successfully completed! :D ")
except:
    notify_slack("Crashed during TFIDF")

