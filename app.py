#import libraries

import uvicorn
from fastapi import FastAPI, Form
import pickle
from pydantic import BaseModel
from typing import List

#from flask import render_template

from fastapi import  Request
from fastapi.templating import Jinja2Templates

#Creating the app object
app = FastAPI()
templates = Jinja2Templates(directory="templates")

#pickle_in = open("rf_model.pkl", "rb")
#rf_model = pickle.load(pickle_in)

# Load the pickle file containing multiple models
with open('ml_multiple_models.pkl', 'rb') as file:
    ml_loaded_models = pickle.load(file)
    
# Load the pickle file containing multiple models
# with open('dl_multiple_models.pkl', 'rb') as file:
#     dl_loaded_models = pickle.load(file)

# Retrieve a specific model by key
rf_model_loaded = ml_loaded_models['rf_model']
#lr_model_loaded = ml_loaded_models['lr_model']
#dt_model_loaded = ml_loaded_models['dt_model']

#fnn_model_loaded = dl_loaded_models['fnn_model']
#cnn_model_loaded = dl_loaded_models['cnn_model']
#rnn_model_loaded = dl_loaded_models['rnn_model']

class creditvalues(BaseModel):
    Age: List[float]
    Income: List[float]
    Home: List[float]
    Emp_length: List[float]
    Intent: List[float]
    Amount: List[float]
    Rate: List[float]
    Percent_income: List[float]
    Default: List[float]
    Cred_length: List[float]

# @app.get('/')

# def index(request: Request):
#     #return render_template('index.html')
#     return templates.TemplateResponse("index.html", {"request": request, "title": "My Page", "content": "Welcome to the Credit Risk Analysis"})
    
@app.get('/')
def predict(request: Request):
    return templates.TemplateResponse("prediction_form.html", {"request": request})
    
#data:creditvalues
@app.post('/predict')
def predict_creditrisk(Age: List[float] = Form(...),
    Income: List[float] = Form(...),
    Home: List[float] = Form(...),
    Emp_length: List[float] = Form(...),
    Intent: List[float] = Form(...),
    Amount: List[float] = Form(...),
    Rate: List[float] = Form(...),
    Percent_income: List[float] = Form(...),
    Default: List[float] = Form(...),
    Cred_length: List[float] = Form(...),
    ):
    # data = data.dict()
    # Age = data['Age']
    # Income = data['Income']
    # Home = data['Home']
    # Emp_length = data['Emp_length']
    # Intent = data['Intent']
    # Amount = data['Amount']
    # Rate = data['Rate']
    # Percent_income = data['Percent_income']
    # Default = data['Default']
    # Cred_length = data['Cred_length']
    
    # Reshape the input data to a 2D array
    input_data = list(zip(Age, Income, Home, Emp_length, Intent, Amount, Rate, Percent_income, Default, Cred_length))
    

    rf_prediction = rf_model_loaded.predict(input_data)
    #lr_prediction = lr_model_loaded.predict(input_data)
    #dt_prediction = dt_model_loaded.predict(input_data)
    #rnn_prediction = rnn_model_loaded.predict(input_data)
    
    return {'rf_prediction': rf_prediction.tolist()}
    #return {'rf_prediction': rf_prediction.tolist(),'rnn_prediction': rnn_prediction.tolist()}

#Run the API with uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port =8000)