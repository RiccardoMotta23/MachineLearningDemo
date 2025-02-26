# --- IMPORT SECTION ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # to standardize (scale) the feature (the X)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # To evaluate the model
import seaborn as sns
from sklearn.datasets import load_iris
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
# Importing the dataset: the Iris dataset contains data of three species of flowers
dataset = load_iris()

# Creating the DataFrame
data = pd.DataFrame(data = dataset.data, columns = dataset.feature_names) # The features and the target (a.k.a the X and the y)
data['target'] = dataset.target # the target (a.k.a the y)

# visualizing the first rows of the dataset
print(f"\nHere are the first 5 rows of the dataset:\n{data.head()}")

# Separate the data in features and target
X = data.iloc[:, :-1].values # all the columns expect the last one
y = data['target'].values # the last column

# Splitting the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101, stratify = y)
# note: the 'stratify' parameter ensures that classes are well balanced between train and test

# Feature scaling
scaler = StandardScaler()
# we are going to scale ONLY the features (i.e. the X) NOT the y
X_train_scaled = scaler.fit_transform(X_train) # fitting to X_train and transforming then
X_test_scaled = scaler.transform(X_test)

# --- END OF MAIN CODE ---