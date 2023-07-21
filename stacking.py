import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_and_preprocess_data(pd_frames):
    tfidf_vectorizer = TfidfVectorizer(dtype='float32')
    tfidf_values = tfidf_vectorizer.fit_transform(pd_frames['sentence'])
    pd_frames.drop(columns=['sentence'], inplace=True)
    pd_frames['success attack (impact/effect)'] = pd_frames['success attack (impact/effect)'].astype(bool)
    pd_frames['type / status'] = pd_frames['type / status'].astype(int)
    return pd_frames['type / status'], tfidf_values


def stacking_method(frames):
    Y, X = load_and_preprocess_data(pd_frames=frames)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    base_models = [
        ('decision_tree', DecisionTreeClassifier()),
    ]
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    # Calculate Precision score
    precision = precision_score(y_test, y_pred, average='macro')
    print("Precision:", precision)
    recall = recall_score(y_test, y_pred, average='macro')
    print("Recall:", recall)
    conf_matrix = confusion_matrix(y_test, y_pred)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    print("True Positives (TP):", TP)
    print("True Negatives (TN):", TN)
    print("False Positives (FP):", FP)
    print("False Negatives (FN):", FN)

