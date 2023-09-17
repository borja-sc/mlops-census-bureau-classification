# Script to train machine learning model.
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, get_slice_performance
import pickle
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == '__main__':
    # Add code to load in the data.
    data = pd.read_csv('data/cleaned_census.csv')

    # Train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True)

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Get model performance.
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    logging.info("Training metrics: {}".format(metrics))

    # Get performance by data slices
    slice_acc = {}
    for feat in cat_features:
        slice_acc[feat] = get_slice_performance(
            train, X_train, y_train, feat, model)
    # Save slice metrics output
    logging.info("Saving slicing metrics")
    with open('slice_output.txt', 'w') as f:
        json.dump(slice_acc, f)
