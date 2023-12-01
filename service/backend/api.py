from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import pickle
import numpy as np

# importing the model
with open(os.path.join(r"model", 'ml_multiple_models.pkl'), 'rb') as file:
    ml_loaded_models = pickle.load(file)

# Retrieve a specific model by key
rf_model_loaded = ml_loaded_models['rf_model']

# crating app
api = FastAPI()  # define app using Flask

class Debtor(BaseModel):
    Age: float
    Income: float
    Home: float
    Emp_length: float
    Intent: float
    Amount: float
    Rate: float
    Percent_income: float
    Default: float
    Cred_length: float

@api.post('/predict')
def predict(debtor: Debtor):    
    
    # Reshape the input data to a 2D array
    input_data = [debtor.Age, debtor.Income, debtor.Home, debtor.Emp_length, debtor.Intent, debtor.Amount, \
        debtor.Rate, debtor.Percent_income, debtor.Default, debtor.Cred_length]
    input_data = np.array(input_data)
    
    rf_prediction = rf_model_loaded.predict(input_data.reshape(1, -1))
    
    return {'rf_prediction': rf_prediction.tolist()[0]} # return a single value
    


#Run the API with uvicorn

if __name__ == '__main__':
    # need to use 0.0.0.0 in docker as localhost to avoid error during startup
    uvicorn.run(api, host='0.0.0.0', port=8080) 