# --- IMPORT SECTION ---
import math

import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
import seaborn as sns
# importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split # to perform train-test split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
path_to_data = "Salary_Data.csv"
data = pd.read_csv(path_to_data)

# Visualizing the dataset
print(f"\nHere are the firsts 5 rows of the dataset:\n{data.head()}")

# Separate the data in features and target
X = data['YearsExperience'].values.reshape(-1 , 1)
y = data['Salary'].values.reshape(-1, 1)

# Using a plot to visualize the data
plt.title("Years of Experience vs Salary") # title of the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(X, y, color="red") # actual plot
sns.regplot(data = data, x = "YearsExperience", y = "Salary") # regression line
plt.show() # renderize the plot to show it

# Splitting the dataset into training to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# Checking the train and test size to prove they are 80% and 20% respectively
print(f"\nThe total X size is: {X.shape[0]}")
print(f"The X_train size is: {X_train.shape[0]} and is the {X_train.shape[0]/X.shape[0] * 100} % of the total X")
print(f"The X_test size is: {X_test.shape[0]} and is the {X_test.shape[0]/X.shape[0] * 100} % of the total X")

print(f"\nThe total y size is: {y.shape[0]}")
print(f"The y_train size is: {y_train.shape[0]} and is the {y_train.shape[0]/y.shape[0] * 100} % of the total y")
print(f"The y_test size is: {y_test.shape[0]} and is the {y_test.shape[0]/y.shape[0] * 100} % of the total y")

# Visualizing data before scaling
print(f"\n-- BEFORE SCALING -- X_train:\n{X_train[:5]}")
print(f"\n-- BEFORE SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- BEFORE SCALING -- X_test:\n{X_test[:5]}")
print(f"\n-- BEFORE SCALING -- y_test:\n{y_test[:5]}")


# Feature scaling
scaler = StandardScaler()
# we are going to scale ONLY the features (i.e. the X) and NOT the y!
X_train_scaled = scaler.fit_transform(X_train) # fitting to X_train and transforming then
X_test_scaled = scaler.transform(X_test)

# Visualizing data after scaling
print(f"\n-- AFTER SCALING -- X_train:\n{X_train_scaled[:5]}")
print(f"\n-- AFTER SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- AFTER SCALING -- X_test:\n{X_test_scaled[:5]}")
print(f"\n-- AFTER SCALING -- y_test:\n{y_test[:5]}")

# Linear Regression
model = LinearRegression()
# perform the training on the train data (i.e. X_train_scaled, y_train)
model.fit(X_train_scaled, y_train)

# predicting new values
y_pred = model.predict(X_test_scaled)

# visualizing the parameters for the Regressor after the training
print(f"\nAfter the training, the params for the Regressor are: {model.coef_}") # the coefficient of the model

# Visualizing the regression
plt.title("Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.scatter(X_test, y_test, color ="red", label = "Real data")
plt.plot(X_test, y_pred, color = "Blue", label = "Predicted Data")
plt.legend()
plt.show()

# Evaluating the model
rmse = math.sqrt(mean_squared_error(y_test, y_pred)) # Root mean squared error
print(f"\nRMSE: {rmse:.2f}")

# --- END OF MAIN CODE ---