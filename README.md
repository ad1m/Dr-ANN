# Dr. ANN
## [https://www.doctorann.xyz](https://www.doctorann.xyz)


A web based system that uses deep learning to diagnosis patients based on their clinical notes.

###Authors - Ravish Chawla, Adam Lieberman, Garrett, Mallory

## Project Structure

#### IPython Notebooks

- data_cleaning.ipynb
	- In this notebook, most of the data loading and pre-processing was done.
- MultiClass_LSTM.ipynb
	- In this notebook, a LSTM was generated and trained on the data generated in the previous notebook.
- metrics.ipynb
	- In this notebook, Metrics were measured on the results from the LSTM.


#### Stored Objects and Models

- cleaned_tfidf_vectorizer_fit
	- This is a JobLib file of our Trained Vectorizer
- icd9_mapping
	- This is a JobLib file of our ICD9-to-Feature Id mapping
- khot_LSTM_.h5
	- This is a Trained neural network on 32 Neurons with the default Learning rate.
- khot_LSTM_1231.h5
	- This is a Trained neural network on 128 Neurons with the default Learning rate.
- khot_LSTM_1353.h5
	- This is a Trained neural network on 128 Neurons and a smaller than default Learning rate.


#### Data Files

- ICD10_Formatted.csv
	- This is a CSV files of ICD-9 to ICD-10 mappings with descriptions.
- nltk
	- This is a list of Stop Words from the NLTK library.

#### Python Files

- get_labels.py
	 - This is a Python class that contains code from the Data-Cleaning Notebook for generating Labels encoded representations.

#### Reports

- project_proposal
	 - This folder contains our Project Proposal along with referenced images and Latex file.
- paper.pdf
	- This is our final Report

#### .GIT files

- .gitattributes
	- Git Large-File system attributes.
- .gitignore
	- Contains list of files to ignore when making commits.
- .gitmodules
	- Contains Submodule links for our repo.


## To run the Code

You can use the IPython notebooks. You will need to obtain the required datasets from MIMIC III, and place them in the same directory. To obtain the pre-processed data, run the Data-Cleaning notebook, use the outputs from it to run the Mulit-Class-LSTM Notebook. Finally, you can run the Metrics notebook to see how them model performed.
