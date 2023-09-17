from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    res = client.get("/")

    assert res.status_code == 200
    assert res.json()[0] == "Welcome to the Census Bureau dataset API!"


def test_post_more_than_50k():

    body = {
        'age': 47,
        'workclass': 'Private',
        'fnlwgt': 198456,
        'education': 'Masters',
        'education-num': 14,
        'marital-status': 'Divorced',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 85000,
        'capital-loss': 1000,
        'hours-per-week': 60,
        'native-country': 'United-States'
    }

    res = client.post("/inference", json=body)
    assert res.status_code == 200
    assert isinstance(res.json()["predictions"], list)
    assert res.json()["predictions"][0] == ">50K"


def test_post_less_than_50k():

    body = {
        'age': 19,
        'workclass': 'Private',
        'fnlwgt': 220437,
        'education': 'HS-grad',
        'education-num': 8,
        'marital-status': 'Never-married',
        'occupation': 'Handlers-cleaners',
        'relationship': 'Not-in-family',
        'race': 'Asian-Pac-Islander',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 2000,
        'hours-per-week': 25,
        'native-country': 'Peru'
    }

    res = client.post("/inference", json=body)

    assert res.status_code == 200
    assert isinstance(res.json()["predictions"], list)
    assert res.json()["predictions"][0] == "<=50K"


def test_post_invalid_input():

    body = {
        'invalid_field_a': 42,
        'invalid_field_b': 'hello world'
    }

    res = client.post("/inference", data=body)
    assert res.status_code != 200
