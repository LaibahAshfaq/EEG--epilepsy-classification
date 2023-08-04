# EEG Epilepsy Classification 

<img width="1079" alt="Screenshot 2023-08-04 at 12 42 18 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/2a9df6bc-d236-4028-8b5e-93442be22ebc">

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

<img width="414" alt="Screenshot 2023-08-04 at 1 08 50 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/33745543-3fee-49fd-8419-925f91413f65">

In a normal EEG (Electroencephalogram) recording, certain features like spikes may be present, but their appearance and frequency vary depending on the context and the state of the brain activity being recorded. Spikes in a normal EEG are generally brief and can be seen in different brain states. because the EEGs for non-epilepsy are scanned when a person could be having their eyes open, or closed , its difficult to say if these spikes are caused by abnormal activity. acrdoding to our research however these oscilattions are most likely background noise and we know isnt showing epileptic seizure activity.

<img width="412" alt="Screenshot 2023-08-04 at 1 09 07 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/5f636d85-5ba0-4753-9c5d-a79fe5d2d6e6">

In an EEG recording of a person with epilepsy, it is common to observe a reduction in the normal background rhythmic activity, resulting in fewer peaks or slower frequency content. This reduction in peaks is often associated with abnormal brain activity and is an essential characteristic used in the diagnosis and monitoring of epilepsy.

Specifically, during an epileptic seizure  the brain's electrical activity may exhibit various abnormal patterns, such as:

Spike-and-Wave Complexes: These are characterized by sharp spikes or spikes followed by slow waves. They are a hallmark of seizures.

Single sharp waves, which are  high-amplitude events.

Slow Waves: Generalized or focal slow waves, which indicate a reduction in normal brain activity.

we can see this here, in the three scans  we see a reduction in activity with slow wave forms, and som high amplitude spikes, and in the average of all epileptic scans we see lots of sharp spike activity. our feature extractor can numerically show these oscilattions and patterns, which helps us classify EEG scans using various models. 

<img width="589" alt="Screenshot 2023-08-02 at 4 10 15 PM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/f7055f5d-2448-42d4-9abf-7318acb3f227">

<img width="347" alt="Screenshot 2023-08-04 at 1 11 49 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/ae108c0d-8369-4fae-87fc-178b5017ea2b">

# Feature Extraction
TSFRESH also known as  (Time Series Feature Extraction on Basis of Scalable Hypothesis) thats a mouthful is a package that extracts a wide range of features from time series data.  Here are some of the features that TSFRESH can extract:
Spectral entropy is a measure of signal irregularity. is often used in various applications, such as speech recognition, music analysis, and biomedical signal processing (e.g., analyzing EEG or ECG signals). 

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

These are the main models we made to classify our data. 
We started off simple and used PCA and UMAP which are both dimensionality reduction techniques. which we'll discuss in the next slide

we used the f1 scores to compare the model's ability to predict accurately because they're best used where there is an imbalanced class distribution in the dataset and The F1 score is a single metric that takes into account both precision and recall, making it a suitable evaluation metric for binary classification tasks. 

Another reason is that false positives and false negatives hold the same consequences in this project. For a medical tool used to classify a serious brain condition, we wouldn't want to misclassify patients with a seizure when they don't have it and classify them to not have a seizure when their scans are showing seizure activity. when making diagnostic classification models its important to consider the gravity of misclassification.

two of our best models are the simple logistic regression with UMAP , a dimensionality reduction tool, and the XGBoost with PCA, another dimensionality reduction tool. 

both tools are really important and have similar outcomes, and the next slide shows how they compare. 
<img width="786" alt="Screenshot 2023-08-04 at 1 12 23 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/f84156e4-fa6f-4f27-95d0-12ae6673e5cc">

# PCA vs UMAP
<img width="511" alt="Screenshot 2023-08-02 at 1 11 41 PM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/2179a904-3100-4408-b468-2efb956c1215">
<img width="601" alt="Screenshot 2023-08-02 at 1 11 04 PM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/a1898a53-d05e-43e9-b67e-83fec51f5aab">
<img width="718" alt="Screenshot 2023-08-02 at 1 10 54 PM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/5a0783a2-d4e9-4b3a-80cb-9f2bff0ae21c">
<img width="315" alt="Screenshot 2023-08-04 at 1 12 40 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/9830c074-095c-4a8e-8985-7d4dfc167007">

# Best Model
Finally, we decided our best model to be the logistic regression with UMAP, based on the scores and how we felt uMAP was a better dimensionality reduction tool for our purposes.

 We got high accuracy, recall, precision, and f1 scores. our scores are fairly high, near perfect which we are very excited about, and plan to use new data to test its accuracy in the next phase of the project. 
 
<img width="728" alt="Screenshot 2023-08-04 at 1 13 08 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/8a7cb28f-84f1-4273-8c0f-0cf626a537ac">

<img width="710" alt="Screenshot 2023-08-04 at 1 14 21 AM" src="https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/9a09b4ea-e2a4-43ac-a498-90b1309ee5c0">


# Model Deployment 

https://github.com/LaibahAshfaq/EEG--epilepsy-classification/assets/128645674/2ed86152-eb7d-4d3e-a17c-70e26ec4e0fc


# Next Steps
for our next steps we want to use domain specific packages for feature extraction, like we discussed before, so we have the most relevant features relating to EEG scans.

we would also like to create a convolutional neural network to image classify actual scans, and not just the features of them.

another step would be to use other data to classify different disorder that use EEG scans as a diagnostic tool, like sleep disorders, and make it a mulivariate classification model.

and finally we want to deploy a more complex app that can classify multiple disorders based on a single EEG scan. this would be invaluable to medical professionals and the neurology profession. 
