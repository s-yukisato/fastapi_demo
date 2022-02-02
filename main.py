import pickle
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

models = pickle.load(open("model.pickle", "rb"))


class Predict(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str


def _titanic_predict(df):
    df["IsAlone"] = ((df["SibSp"] == 0) & (df["Parch"] == 0)).astype(int)
    df["Embarked"].fillna(("S"), inplace=True)
    df["Age"].fillna((df["Age"].median()), inplace=True)
    df["Fare"].fillna((df["Fare"].mean()), inplace=True)
    df = pd.get_dummies(df, columns=["Embarked", "Sex"])
    delete_columns = ["Name", "PassengerId", "Ticket", "Cabin"]
    df.drop(delete_columns, axis=1, inplace=True)

    score = []
    for gbm in models:
        score.append(gbm.predict(df))

    proba = np.mean(score)

    return proba


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/titanic/")
def titanic(req: Predict):
    df = pd.DataFrame([req.dict()])
    prediction = _titanic_predict(df)
    return {"res": "ok", "proba of survive": prediction}
