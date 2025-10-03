# AIM: Implement Decision-Tree learning algorithm using Entropy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Function to import data
def importdata():
    # Make sure the CSV file is in the same folder as this script
    balance_data = pd.read_csv("balance-scale.data", header=None)  # no header in file

    print("Dataset Length:", len(balance_data))
    print("Dataset (first 5 rows):")
    print(balance_data.head())
    return balance_data

# Function to split dataset into training and testing sets
def splitdataset(balance_data):
    X = balance_data.values[:, 1:5]  # features
    Y = balance_data.values[:, 0]    # target label

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100
    )
    return X, Y, X_train, X_test, y_train, y_test

# Function to train decision tree classifier using entropy
def train_using_entropy(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5
    )
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Function to make predictions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted Values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Main function
def main():
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    
    clf_entropy = train_using_entropy(X_train, y_train)
    
    print("\nResults using Entropy:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

if __name__ == "__main__":
    main()
