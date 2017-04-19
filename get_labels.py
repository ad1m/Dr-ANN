import pandas as pd;
import numpy as np;
import scipy as sp;
import re;
import pickle;
import os;

'''
Loads in the Dataset from CSV files and places them in a Pandas Dataframe.
'''
def load_data():
    diagnosis = pd.read_csv('/home/ubuntu/workspace/data/DIAGNOSES_ICD.csv');
    notes = pd.read_csv('/home/ubuntu/workspace/data/NOTEEVENTS.csv');
    
    return diagnosis, notes;

'''Drop rows with NA values'''

'''
Takes a String ICD9 code and returns it's truncated value.
'''
def format_icd9(icd9):
    if icd9[0] == "V":
        return icd9[0:3]
    if icd9[0] == "E":
        return icd9[0:4]
    else: 
        return icd9[0:3]

#Truncated ICD9s are obtained for each row.
'''
Creates a K-Hot encoded vector from a list of ICD9 \
codes using a dictionary mapping.
'''
def feature_mapping(icd9_mapping, icd9_set):
    vector = np.zeros((len(icd9_mapping)))
    for icd9 in icd9_set:
        vector[icd9_mapping[icd9]] = 1
    return vector

def get_labels():
    diagnosis, notes = load_data();

    diagnosis = diagnosis.dropna(axis=0, how="any")

    groups = notes.groupby('HADM_ID').apply(lambda row: list(set(row['TEXT'])));

    formatted_icd9_codes = diagnosis["ICD9_CODE"].apply(format_icd9)

    '''
    Attach the Formatted ICD9 codes as a column to the Diagnosis table.
    '''
    diagnosis_reduced_icd9 = diagnosis.join(formatted_icd9_codes, lsuffix="_l", rsuffix="_r")
    diagnosis_reduced_icd9 = diagnosis_reduced_icd9[["HADM_ID", "ICD9_CODE_r"]]
    diagnosis_reduced_icd9.columns = ["HADM_ID", "ICD9_CODE"]

    '''
    Create a SET ov ICD9 code values for each Hospital visit in Diagnosis.
    '''
    diagnosis_group_reduced = diagnosis_reduced_icd9.groupby('HADM_ID').apply(lambda x: set(x['ICD9_CODE']));
    diagnosis_count_reduced = diagnosis_group_reduced.apply(lambda x: len(x))
    diagnosis_group_reduced.name = "ICD9_set"

    '''
    Attach the SET of ICD9 code values as a column to the Notes table.
    '''
    notes_icd9 = notes.set_index("HADM_ID").join(diagnosis_group_reduced, how="inner", lsuffix="_l", rsuffix="_r")
    notes_icd9 = notes_icd9[["TEXT", "ICD9_set"]]

    icd9_mapping = dict(zip( set(diagnosis_reduced_icd9["ICD9_CODE"].values) , np.arange(0, len(set(formatted_icd9_codes))) ))
    notes_icd9["vector"] = notes_icd9["ICD9_set"].apply(lambda x: feature_mapping(icd9_mapping, x));
    labels = notes_icd9["vector"].values

    return labels;
