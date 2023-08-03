import streamlit as st 
import pandas as pd 

from sklearn.base import BaseEstimator, TransformerMixin

from tsfresh import defaults
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.transformers.feature_augmenter import FeatureAugmenter
from tsfresh.transformers.feature_selector import FeatureSelector

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer

#imports for Logistic Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from tsfresh.transformers import RelevantFeatureAugmenter
import umap
import pickle


st.title('Epilepsy Classification')
st.header('upload CSV file')

uploaded_file = st.file_uploader('Upload a file')

# Load Model
loaded_model = pickle.load(open("pca_logreg_model.pickle", "rb"))
        
if st.button('predict'):
    pred_data = pd.read_csv(uploaded_file)
    
    #Plot this data     
    original_data = pd.read_csv('data/Seizure_data.csv')
    row_to_plot = original_data[original_data['Unnamed'] == pred_data['Unnamed'][0]]
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.xticks(rotation=45, ha='right', fontsize = 0)
    # Plot the row data
    plt.plot(row_to_plot.T.values[1:])
    plt.xlabel('Time', fontsize = 15)
    plt.ylabel('Value of EEG scan')
    plt.title('EEG Scan')
    st.pyplot(fig)

    
    #Predict this data
    feature_data = pd.read_csv('data/extracted_features.csv')
    pred_data_features = feature_data[feature_data['Unnamed: 0'] == pred_data['Unnamed'][0]]
    
    pred_y = loaded_model.predict(pred_data_features.drop(columns = ["Unnamed: 0"]))
    if pred_y ==1:
        st.write("EEG Scan shows Epileptic Seizure")
    else: 
        st.write("EEG Scan doesn't show Epilepsy")