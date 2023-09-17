import pandas as pd
import numpy as np
import pytest

from starter.ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def fake_data():
    """
    Fake data for model training
    """
    df = pd.DataFrame({
        "var1": [1, 2, -3, -1, 2, 3],
        "var2": [0, 0, 0, 1, 1, 1],
        "var3": [2.7, 1.5, -0.8, 0.2, -2, 0.3],
        "label": [1, 1, 1, 1, 0, 0]
    })

    return df


def test_train_model(fake_data):
    """
    Tests if a model can be correctly trained
    """

    X_fake = fake_data.copy()
    y_fake = X_fake.pop("label")

    model = train_model(X_fake, y_fake)

    assert len(model.classes_) == 2


def test_compute_model_metrics():
    """
    Compute metrics with fake arrays
    """
    fake_y = np.array([1, 1, 1, 0, 0, 1, 1, 1])
    fake_preds = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    metrics = compute_model_metrics(fake_y, fake_preds)

    assert all([metrics["precision"] == 0.75, metrics["recall"] == 0.5, metrics["fbeta"] == 0.6])


def test_inference(fake_data):
    """
    Compute inference with fake arrays
    """
    X_fake = fake_data.copy()
    y_fake = X_fake.pop("label")
    model = train_model(X_fake, y_fake)

    assert all(inference(model, X_fake) == [1, 1, 1, 1, 0, 0])
