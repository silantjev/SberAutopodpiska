import os
import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


path_dir = os.path.dirname(__file__)
path = os.path.join(path_dir, 'dill_pipe.pkl')
with open(path, 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_brand: str
    device_os: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    predition: int


app = FastAPI()


@app.get('/status')
def status():
    return 'Active'


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    X = pd.DataFrame.from_dict([form.dict()])

    pipe = model['pipe']
    proba = pipe.predict_proba(X)[:, 1]
    threshold = 0.03
    predict = (proba > threshold) #.astype(int)
    
    return {'predition': int(predict[0])}
