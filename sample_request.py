import requests

body = {
    'age': 38,
    'workclass': 'Private',
    'fnlwgt': 215646,
    'education': 'HS-grad',
    'education-num': 9,
    'marital-status': 'Divorced',
    'occupation': 'Handlers-cleaners',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}

res = requests.post("https://mlops-census-api.onrender.com/inference", json=body)

print("Status code:", res.status_code)
print("Response:", res.json())