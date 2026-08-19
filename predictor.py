import joblib
import pandas as pd

pipeline = joblib.load("model/pipeline.pkl")

def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = pipeline.predict(df)[0]

    probability = pipeline.predict_proba(df)[0].tolist()

    if probability[0] > 0.5:

        return{
            "prediction": int(prediction),
            "verdict": "This passenger will not survive",
            "probability":{
                "No Survival": probability[0],
                "Survival": probability[1]
            }
        }
    else:

        return{
            "prediction": int(prediction),
            "verdict": "This passenger will survive",
            "probability":{
                "No Survival": probability[0],
                "Survival": probability[1]
            }
        }