
# coding: utf-8

# In[58]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd;
import numpy as np;
import scipy as sp;
import re;
import pickle;
import os;
import subprocess
import matplotlib.pyplot as plt

import sklearn;
from sklearn.utils import shuffle;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;
from sklearn.decomposition import TruncatedSVD


# In[4]:

def notify_slack(text):
    subprocess.Popen('''curl -X POST --data-urlencode "payload={'channel' : '#random', 'username': 'webhookbot', 'text':'''+ '\'' + text + '\'' + '''}" https://hooks.slack.com/services/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''', shell=True)
 


# This function will read in data to be vectorized for the LSTM model

# In[73]:

notify_slack("Starting data load")

def load_data():
    diagnosis = pd.read_csv('/home/ubuntu/workspace/data/DIAGNOSES_ICD.csv');
    notes = pd.read_csv('/home/ubuntu/workspace/data/NOTEEVENTS.csv');
    
    return diagnosis, notes;

diagnosis, notes = load_data();
diagnosis = diagnosis.dropna(axis=0, how="any")


# In[41]:

groups = notes.groupby('HADM_ID').apply(lambda row: list(set(row['TEXT'])));


# In[74]:

print ("Overall: ", len(diagnosis))
print ("Unique codes:", len(set(diagnosis["ICD9_CODE"].values)))

# Takes string and returns a formated icd9 code
def format_icd9(icd9):
    if icd9[0] == "V":
        return icd9[0:3]
    if icd9[0] == "E":
        return icd9[0:4]
    else: 
        return icd9[0:3]

formatted_icd9_codes = diagnosis["ICD9_CODE"].apply(format_icd9)
print ("Reduced unique codes", len(set(formatted_icd9_codes)))


# In[75]:

diagnosis_reduced_icd9 = diagnosis.join(formatted_icd9_codes, lsuffix="_l", rsuffix="_r")
diagnosis_reduced_icd9 = diagnosis_reduced_icd9[["HADM_ID", "ICD9_CODE_r"]]
diagnosis_reduced_icd9.columns = ["HADM_ID", "ICD9_CODE"]


# In[84]:

diagnosis_group = diagnosis.groupby('HADM_ID').apply(lambda x: list(x['ICD9_CODE']));
diagnosis_group_reduced = diagnosis_reduced_icd9.groupby('HADM_ID').apply(lambda x: set(x['ICD9_CODE']));

diagnosis_count = diagnosis_group.apply(lambda x: len(x))
diagnosis_count_reduced = diagnosis_group_reduced.apply(lambda x: len(x))

diagnosis_count.hist(bins=35)
plt.show()

diagnosis_count_reduced.hist(bins=35)
plt.show()


# In[124]:

#print(diagnosis_group_reduced)
diagnosis_group_reduced.name = "ICD9_set"
notes_icd9 = notes.set_index("HADM_ID").join(diagnosis_group_reduced, how="inner", lsuffix="_l", rsuffix="_r")
notes_icd9 = notes_icd9[["TEXT", "ICD9_set"]]
print(notes_icd9.shape)


# In[138]:

icd9_mapping = dict(zip( set(diagnosis_reduced_icd9["ICD9_CODE"].values) , np.arange(0, len(set(formatted_icd9_codes))) ))

# indexing into dict to get feature array mapping
# takes in a icd9 set object
def feature_mapping(icd9_set):
    vector = np.zeros((len(icd9_mapping)))
    for icd9 in icd9_set:
        vector[icd9_mapping[icd9]] = 1
    return vector

#drop 47 notes corresponding to 47 dropped diagnosis? 

notes_icd9["vector"] = notes_icd9["ICD9_set"].apply(feature_mapping)
print (notes_icd9.shape)


# In[148]:

print(notes_icd9.iloc[0])
print(len(notes_icd9["vector"].iloc[0]))
labels = notes_icd9["vector"].values
print(labels[0].shape)
print (sum(labels[0]))
pickle.dump(labels, open("/mnt/labels", "wb"))
os.exit()


# Next, we create a custom stop words method.

# In[19]:

def get_stopwords():
    stop_words = str();
    with open('nltk', 'r') as f:
        for line in f:
            stop_words = stop_words + '\\b' + line.strip() + '\\b' + '|';

    stop_words = stop_words[0:-1];

    return stop_words;


# We create a function that takes in a textual clinical note and run preprocessing steps on it to remove dates, lower case all the words, etc. 

# In[20]:

def clean_text(notes_df):
    stop_words = get_stopwords();
    
    # Need to remove single charcater items
    # Need to remove the leading 0 from a digit (i.e. 07 = 7) -OR- replace with "DIGIT"
    # replace dates with "DATE" or something 
    
    
    #notes_filtered = notes_df['TEXT'].apply(lambda row: re.sub("[^a-zA-Z0-9]", " ", row.lower()));
    notes_filtered = notes_df['TEXT'].apply(lambda row: re.sub("21[0-9]{2}.[0-1]?[0-9]{1}.[0-3]?[0-9]{1}.+[0-2]{1}[0-9]{1}:[0-5]{1}[0-9]{1}.+[\bAM\b|\bPM\b]", " ", row));
    notes_filtered = notes_filtered.apply(lambda row: re.sub("[^a-zA-Z0-9\.]", " ", row.lower()));
    #notes_filtered = notes_filtered.apply(lambda row: re.sub("\W\d+\.?\d*", "DIGIT", row));
    #notes_filtered = notes_filtered.apply(lambda row: re.sub("\s[a-zA-Z]\s", " ", row));                                        

    notes_nostops = notes_filtered.apply(lambda row: re.sub(stop_words, ' ', row));
    
    notes_final = notes_nostops.apply(lambda row: " ".join(row.split()));
    
    notes_df = notes_df.drop('TEXT', axis=1);
    notes_df = notes_df.assign(TEXT = notes_final.values)
    
    return notes_df;


# In[21]:

#notes_filtered = clean_text(notes);
#pickle.dump(notes_filtered, open('/home/ubuntu/CliNER/data/saved/notes_filtered', "wb"), protocol=2);
notes_filtered = pickle.load(open('/home/ubuntu/CliNER/data/saved/notes_filtered','rb'));
#notes_filtered = notes_filtered[0:10000]


# A method to parse the output from CliNER into a vocabulary. We use this vocab to inform out TFIDF Transformer in order to reduce our vector size from millions down to 64k important words. 

# In[22]:

notify_slack("Getting CliNER vocab")

def get_cliner_vocab():
    text_list = []
    type_list = []
    for f in os.listdir('/home/ubuntu/CliNER/data/CLEANED/'):
        with open('/home/ubuntu/CliNER/data/CLEANED/' + f, 'rb') as file:
            for line in file:
                matches_text = re.search('(?<=c=\").*?(?=\" )', line); # gets the highlighted text
                matches_text_group0 = re.sub("[^a-zA-Z0-9]", " ", matches_text.group(0))
                matches_type = re.search('(?<=t=\").*?(?=\")', line); # gets the designation
                text_list.append(matches_text_group0)
                type_list.append(matches_type.group(0))
    
    return text_list, type_list


# In[23]:

text_list, type_list = get_cliner_vocab()
max_length = max([len(text.split()) for text in text_list])
min_length = min([len(text.split()) for text in text_list])


# In[129]:

#cleaned_data_vectorizer = CountVectorizer(ngram_range=(min_length, max_length), stop_words='english');

#cleaned_vectorizer_updated = cleaned_data_vectorizer.fit(text_list);

#cleaned_notes_counts = cleaned_vectorizer_updated.transform(text_list);


# In[56]:

cleaned_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english', min_df=1);
cleaned_tfidf_vectorizer_updated = cleaned_tfidf_vectorizer.fit(text_list);
print (len(cleaned_tfidf_vectorizer_updated.vocabulary_))
print (notes_filtered["TEXT"].shape)


# In[25]:

try:
    notify_slack("Starting TFIDF fit & transform")
    cleaned_tfidf_vectorizer_updated = cleaned_tfidf_vectorizer.fit(text_list);
    cleaned_tfidf_counts = cleaned_tfidf_vectorizer_updated.transform(notes_filtered["TEXT"]);
    
    notify_slack("Pickling sparse counts into current directory")
    pickle.dump(cleaned_tfidf_counts, open("sparse_tfidf_min_df=0.05",'wb'))
    
    notify_slack("Fitting with Truncated SVD")
    svd = TruncatedSVD(n_components = 1000)
    clean_tfidf_reduced = svd.fit_transform(cleaned_tfidf_counts)
    
    notify_slack("Pickling array into current directory")
    pickle.dump(clean_tfidf_reduced, open("tfidf_vectors_min_df=0.05",'wb'))
    
    notify_slack("Pickling SVD into current directory")
    pickle.dump(svd, open('fit_svd_model_min_df=0.05', 'wb'))
    
    notify_slack("Successfully completed! :D ")
    
except:
    notify_slack("Crashed during TFIDF")


# ### Saved for  potential reuse
# ---
# vectorizer = CountVectorizer();
# 
# vectorizer_updated = vectorizer.fit(notes_filtered['TEXT']);
# 
# notes_counts = vectorizer_updated.transform(notes_filtered['TEXT']);
# 
# ### Get top words for the report
# #notes_order = np.argsort(notes_counts.todense(), axis=0);
# 
# notes_sum = np.sum(notes_counts, axis=0);
# 
# notes_sort = np.argsort(-1 * notes_sum[0]);
# 
# name_counts = pd.DataFrame({'name' : vectorizer_updated.vocabulary_.keys(), 'idx' : vectorizer_updated.vocabulary_.values()})
# #name_counts = name_counts.set_index('idx');
# 
# sort_list = list();
# sum_list = list();
# for i in range(notes_sort.shape[1]):
#     sort_list.append(notes_sort[0,i]);
#     sum_list.append(notes_sum[0,i]);
#     
# count_df = name_counts.sort_values(by='idx', ascending=True);
# sum_df = pd.DataFrame({'count' : sum_list, 'idx' : np.arange(0,len(sum_list))});
# 
# count_sum = count_df.join(sum_df, on='idx', lsuffix='c', rsuffix='s').drop('idxs', axis=1);
# count_sum_top = count_sum.sort_values(by='count', ascending=False);
# 
# shuffle(count_sum_top.loc[count_sum_top['count'] == 1])
# 
# name_counts.sort_values(by='count', ascending=False)

# In[ ]:



