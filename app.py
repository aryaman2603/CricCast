import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sklearn

app = FastAPI(title="Cricket Score Prediction API")
print(f"🔎 API SERVER is using Scikit-Learn version: {sklearn.__version__}")
with open("artifacts/model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

class MatchInput(BaseModel):
    venue: str
    batting_team: str
    bowling_team: str
    innings: int
    ball: float
    current_score: int
    wickets_fallen: int
    runs_last_5: int
    wickets_last_5: int

@app.get("/")
def home():
    return {"message": "Welcome to the Cricket Score Prediction API!"}

@app.post("/predict")
def predict_score(data: MatchInput):
    try:
        wickets_left = 10 - data.wickets_fallen
        over_num = int(data.ball)
        ball_num = int((data.ball - over_num) * 10)
        legal_balls_bowled = (over_num * 6) + ball_num
        balls_left = 120 - legal_balls_bowled

        if legal_balls_bowled==0:
            crr=0.0
        else:
            crr = (data.current_score * 6) / legal_balls_bowled

        input_df = pd.DataFrame([{
            'venue': data.venue,
            'batting_team': data.batting_team,
            'bowling_team': data.bowling_team,
            'innings': data.innings,
            'ball': data.ball,
            'legal_balls_bowled': legal_balls_bowled,
            'wickets_left': wickets_left,
            'balls_left': balls_left,
            'current_score': data.current_score,
            'crr': crr,
            'runs_last_5': data.runs_last_5,
            'wickets_last_5': data.wickets_last_5
        }])

        prediction = model_pipeline.predict(input_df)[0]

        return{
            "current_score": data.current_score,
            "predicted_final_score": int(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))