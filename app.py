from fastapi import FastAPI

import uvicorn

from predictor import predict
from schemas import Passenger

app = FastAPI(
    title = "Titanic Survival API",
    version = "1.0"
)

@app.get("/")
def home():

    return {
        "message": "Titanic Prediction API is running"
    }
    
@app.post("/predict")
def predict_survival(passenger: Passenger):

    result = predict(passenger.model_dump())

    return result

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # Points to this file (app) and the variable (app)
        host="127.0.0.1",  # Localhost
        port=8000,  # Port to listen on
        reload=True,  # Enables auto-reload on code changes
    )