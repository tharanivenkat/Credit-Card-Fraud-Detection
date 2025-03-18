import numpy as np
import pandas as pd #Used for numerical operations and data handling.
from sklearn.model_selection import train_test_split  #Helps in splitting data into training/testing sets.
from sklearn.preprocessing import StandardScaler #Used for scaling numerical features.
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM #Machine learning models for anomaly detection.
from sklearn.metrics import classification_report, accuracy_score #Used for evaluation.
import matplotlib.pyplot as plt
import seaborn as sns  #Used for data visualization.
from pylab import rcParams #Used to set figure size.

rcParams['figure.figsize'] = 14, 8   #Defines default figure size for plots.
#Sets a random seed to ensure reproducibility.
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"] #Used for class labeling (0 = Normal, 1 = Fraud)

# Load the dataset
data = pd.read_csv(r"C:\Users\ch.Tharani\Desktop\My Projects\ML Project 1\creditcard.csv")

# Explore the data
data.info() #Shows dataset structure, column types, and memory usage.
print("Missing values:", data.isnull().values.any())
  #Checks if there are missing values.

# Plot transaction class distribution
count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Get the Fraud and the normal dataset
#Plots a bar chart showing class distribution.
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
print("Fraud shape:", fraud.shape, "Normal shape:", normal.shape)

# Explore Amount per transaction by class
#Creates histograms to compare fraud and normal transaction amounts.
#Uses log scale for better visualization.
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud['Amount'], bins=bins)
ax1.set_title('Fraud')
ax2.hist(normal['Amount'], bins=bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()

# Explore Time of transaction vs Amount by class
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud['Time'], fraud['Amount'])
ax1.set_title('Fraud')
ax2.scatter(normal['Time'], normal['Amount'])
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
#Scatter plot shows the relationship between Time and Amount for fraud and normal transactions.

# Sample data and determine the number of fraud and valid transactions
##Takes 10% random sample of data to reduce computation time.
data1 = data.sample(frac=0.1, random_state=1)
print("Sampled data shape:", data1.shape)
print("Original data shape:", data.shape)

#Calculates outlier fraction, which is the proportion of fraud cases.
Fraud = data1[data1['Class'] == 1]
Valid = data1[data1['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print("Outlier fraction:", outlier_fraction)

print("Fraud Cases:", len(Fraud))
print("Valid Cases:", len(Valid))

# Plot correlation matrix
#Creates a heatmap to show correlation between features.
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# Plot heat map
g = sns.heatmap(data1[top_corr_features].corr(), annot=True, cmap="RdYlGn")

# Create independent and Dependent Features
columns = data1.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)

# Model prediction
#Uses Isolation Forest and Local Outlier Factor for anomaly detection.
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X), 
                                        contamination=outlier_fraction, random_state=state, verbose=0),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                               leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, contamination=outlier_fraction),
    
}

n_outliers = len(Fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    # Reshape the prediction values to 0 for Valid transactions, 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    
    # Run Classification Metrics
    print("{}: {}".format(clf_name, n_errors))
    print("Accuracy Score:")
    print(accuracy_score(Y, y_pred))
    print("Classification Report:")
    print(classification_report(Y, y_pred))

    #Loads and explores the credit card dataset.
#Visualizes fraud vs normal transactions.
#Samples 10% of data for efficiency.
#Uses Isolation Forest & LOF for anomaly detection.
#Evaluates models using accuracy and classification reports.