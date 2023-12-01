import uvicorn
from fastapi import FastAPI, Form
import requests
import os

#from flask import render_template

from fastapi import  Request
from fastapi.templating import Jinja2Templates

#Creating the app object
app = FastAPI()
templates = Jinja2Templates(directory="templates")

API_HOST = str(os.getenv("BACKEND_HOST"))
API_PORT = str(os.getenv("BACKEND_PORT"))
    

@app.get('/')
def welcome(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/predict')
def get_predict(request: Request):
    return templates.TemplateResponse("prediction_form.html", {"request": request})
    
@app.post('/predict')
def post_predict(request: Request,
                 Age: float = Form(...),
                 Income: float = Form(...),
                 Home: float = Form(...),
                 Emp_length: float = Form(...),
                 Intent: float = Form(...),
                 Amount: float = Form(...),
                 Rate: float = Form(...),
                 Percent_income: float = Form(...),
                 Default: float = Form(...),
                 Cred_length: float = Form(...)
                 ):
    
    # json input creation: I follow the standard used by the backend service
    # for the col_values I take only the first element because I implemented a single call prediction 
    col_names = ["Age", "Income", "Home", "Emp_length", "Intent", "Amount", "Rate", "Percent_income", "Default", "Cred_length"]
    col_values = [Age, Income, Home, Emp_length, Intent, Amount, Rate, Percent_income, Default, Cred_length]
    json_input = dict(zip(col_names, col_values))
    
    # api call to obtain the result
    api_url = f"http://{API_HOST}:{API_PORT}/predict"
    
    response = requests.post(api_url, json=json_input)
    response = response.json()
    
    return templates.TemplateResponse("prediction_form.html", {"request": request, "rf_prediction": response["rf_prediction"]} )

#Run the API with uvicorn

if __name__ == '__main__':
    # need to use 0.0.0.0 in docker as localhost to avoid error during startup
    uvicorn.run(app, host='0.0.0.0', port=8080)