import pickle
import pandas as pd
import numpy as np
from starter.ml.model import inference
from starter.ml.data import process_data
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.train_model import cat_features


class Data(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    # Schema example
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'age': 27,
                    'workclass': 'Self-emp-inc',
                    'fnlwgt': 384756,
                    'education': 'Bachelors',
                    'education-num': 13,
                    'marital-status': 'Never-married',
                    'occupation': 'Prof-specialty',
                    'relationship': 'Not-in-family',
                    'race': 'Asian-Pac-Islander',
                    'sex': 'Female',
                    'capital-gain': 0,
                    'capital-loss': 0,
                    'hours-per-week': 20,
                    'native-country': 'Japan'
                }
            ]
        }
    }


app = FastAPI(
    title="Census data API",
    description="An API that allows to infer income level from census data.",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {"Welcome to the Census Bureau dataset API!"}


@app.post("/inference/")
async def infer(data: Data):
    # Load model and encoders:
    with open("./model/trainedmodel.pkl", "rb") as file:
        model = pickle.load(file)

    with open("./model/data_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    with open("./model/label_encoder.pkl", "rb") as file:
        lb = pickle.load(file)

    array = np.array([[
                     data.age,
                     data.workclass,
                     data.fnlwgt,
                     data.education,
                     data.education_num,
                     data.marital_status,
                     data.occupation,
                     data.relationship,
                     data.race,
                     data.sex,
                     data.capital_gain,
                     data.capital_loss,
                     data.hours_per_week,
                     data.native_country
                     ]])
    df = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ])
    X, _, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(model, X)
    preds = lb.inverse_transform(preds)
    return {"predictions": preds.tolist()}
