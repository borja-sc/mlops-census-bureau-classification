from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.linear_model import LogisticRegression
import pickle
import logging


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    logging.info("Training model")
    model = LogisticRegression(C=1.0, solver='newton-cg', max_iter=150)
    model.fit(X_train, y_train)
    pickle.dump(model, open('model/trainedmodel.pkl', 'wb'))
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return {"precision": precision, "recall": recall, "fbeta": fbeta}


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logging.info("Calculating predictions")
    preds = model.predict(X)
    return preds


def get_slice_performance(df, X, y, feature, model):
    """ Computes performance metrics for slices of data based on a feature.

    Inputs
    ------
    df : pd.DataFrame
        dataframe of dataset
    X : np.array
        array of dataframe features
    y : np.array
        array of dataframe labels
    model : Object
        model from which to calculate metrics
    Returns
    -------
    slice_acc : dictionary
        dictionary with precision, recall, and fbeta for each slice.
    """
    logging.info("Calculating slice performance metrics")
    slice_acc = {}
    for val in df[feature].unique():
        df_temp = df[df[feature] == val]  # rows with specific class
        y_preds = inference(model, X[df_temp.index.values])
        slice_acc[val] = compute_model_metrics(
            y[df_temp.index.values], y_preds)
    return slice_acc
