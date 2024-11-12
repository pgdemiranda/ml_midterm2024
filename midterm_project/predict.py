import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb

# Load the model and preprocessor from the pickle file
with open('xgboost_model_with_preprocessor.pkl', 'rb') as f:
    model, preprocessor = pickle.load(f)

# FastAPI App
app = FastAPI()

# Define the input data model for the API request
class InputData(BaseModel):
    relationship: str
    marital_status: str
    education: str
    occupation: str
    hours_per_week: int
    capital_loss: int
    capital_gain: int
    age: int
    education_num: int

@app.post("/predict/")
async def predict(data: InputData):
    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Make the prediction directly using the model (which includes preprocessing)
    prediction = model.predict(input_data)

    # Return the predicted value (0 or 1)
    return {"prediction": int(prediction[0])}
