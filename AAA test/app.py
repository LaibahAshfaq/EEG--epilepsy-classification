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






st.title('Epilepsy Classification')


st.header('upload CSV file')

uploaded_file = st.file_uploader('Upload a file')






if st.button('create data'):
        pred_data = pd.read_csv(uploaded_file)
        y = pd.read_csv('y_for_modeling.csv')
        X = pd.read_csv('Data/extracted_features.csv')

        pred_data = extract_relevant_features(X, y , column_id = "Unnamed", column_sort='Time', column_value="Value").fit_transform(pred_data, pred_data['y'])

        
y = pd.read_csv('data/y_for_modeling.csv')

X = pd.read_csv('data/extracted_features.csv')







#df_X_features.to_csv('Data/extracted_features.csv')










#X.rename(columns={"Unnamed: 0": 'Segment'}, inplace=True)

model = LogisticRegression(random_state=42)

if st.button('predict'):
    model.fit(X, y)



    y_pred = model.predict(pred_data)
    if y_pred ==1:
        st.write("you're epileptic AF")
    else: 
        st.write("you're good")
