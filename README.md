# EEG Epilepsy Classification 

<img width="1079" alt="Screenshot 2023-08-04 at 12 42 18 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/2a9df6bc-d236-4028-8b5e-93442be22ebc">

# Overview

# What is Epilepsy

It's a neurological disorder characterized by recurrent, unprovoked seizures.
The cause of epilepsy varies; there can be no cause or caused by brain injuries, infection, tumors, genetics, etc. Diagnosing epilepsy involves:  neurological examination, and often  EEG recordings to detect abnormal brain activity during or between seizures. According to the World Health Organization (WHO), approximately 50 million people worldwide have epilepsy. 

EEG signals record the electrical activity of the brain using EEG electrodes placed on the scalp. 
Doctors, neuroscientists, and biomedical engineers usually receive training for years to understand and extract meaningful information from EEG data. 
Even in these cases, the raw recorded data needs to be processed before specialists look at it.
From a computational point of view, the raw EEG signal is simply a discrete-time multivariate time-series. (usually with multiple dimensions), which is how we evaluate it.

# Business Understanding

AI and machine learning tools are the perfect companion to automate, extend, and improve electroencephalogram (EEG) scan interpretation from medical professionals. 
It takes years of training to accurately be able to interpret these brain scans and we believe machine learning can make this process easier for physicians to diagnose patients with severe brain conditions, like epilepsy. 
Some of our current stakeholders who have funded this project, are Columbia University, and Mount Sinai Department of Neurology

# Dataset
We acquired our data from the UCI Machine learning database.
The original dataset from the reference consists of  a total of 500 individuals with each has 4097 data points for 23.5 seconds. Each data point is the value of the EEG recording at a different point in time

We divided and shuffled every 4097 data points into 23 chunks, now we have 11500 pieces of information(row), each piece of information contains 178 data points for 1 second(column), the last column represents the target, which is multivariate until we made it binary. so values 2-5 are non epileptic and value 1 is. 2-5 classified scans show scans when the individual has variable activity, so they may have their eyes open, or closed or are asleep, but the scans themselves aren't showing epileptic activity. 
<img width="429" alt="Screenshot 2023-08-04 at 1 06 54 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classificatio<img width="425" alt="Screenshot 2023-08-04 at 1 07 13 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/703413f3-cb69-407c-bc44-bb2dd158a9fd">
n/assets/128645674/6ce0f986-a5a7-4dbc-837e-bb173db3289f">

# Feature Extraction
TSFRESH also known as  (Time Series Feature Extraction on Basis of Scalable Hypothesis) thats a mouthful is a package that extracts a wide range of features from time series data.  Here are some of the features that TSFRESH can extract:
Spectral entropyis a measure of signal irregularity. is often used in various applications, such as speech recognition, music analysis, and biomedical signal processing (e.g., analyzing EEG or ECG signals). 

and then there are the none domain-specific features such as local minima/ maxima and the height of peaks, which are relevant to all time series analysis.

our final dataset had 370 features, for each time segment, and the feature extractor calculated the final features based on the p-value. 
During feature extraction in TSFRESH, statistical tests are performed to assess whether a particular feature is statistically significant or not. 
 A low p-value indicates that the feature is unlikely to have occurred by chance, which suggests that it may be relevant for predicting the target variable.

Our next step for Data Acquisition is to use domain-specific packages made for EEG scans, 
Some popular EEG feature extraction tools include:

MNE-Python: MNE-Python is a powerful Python library for analyzing EEG and other neurophysiological data. 

PyEEG: PyEEG is a Python library specifically focused on extracting features from EEG signals. 

NeuroKit2: NeuroKit2 is a Python library for biosignal processing, including EEG.

What led us to use TSFRESH was its comprehensive ability to feature extract a simple multivariate time series and its ease of use. other packages are not as established and aren't easily integrated into the MACHINE learning libraries we used for this project. so for that reason, we used a general extractor instead of a domain-specific one. 

# Models made

# PCA vs UMAP

# Best Model

# Model Deployment 

https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/2ed86152-eb7d-4d3e-a17c-70e26ec4e0fc




# Conclusion
