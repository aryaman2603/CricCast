import pandas as pd
import os
import pickle
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def run(self):
        print("Loading training data...")
        df = pd.read_csv(self.data_path)
        weights = df['sample_weight']
        X = df.drop(columns=['match_id', 'final_score', 'sample_weight'])
        y = df['final_score']

        print("Features used: {list(X.columns)}")

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )

        numeric_features = ['ball','legal_balls_bowled', 'wickets_left', 'balls_left', 
                            'current_score', 'crr', 'runs_last_5', 'wickets_last_5']
        categorical_features = ['venue', 'batting_team', 'bowling_team', 'innings']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1,
            random_state=42
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        mlflow.set_experiment("Cricket Score Prediction")

        with mlflow.start_run():
            print("Training model...")
            pipeline.fit(X_train, y_train, model__sample_weight=w_train)

            print("Evaluating...")
            predictions = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"\n✅ Model Performance:")
            print(f"   MAE: {mae:.2f} runs (Average Error)")
            print(f"   R2 Score: {r2:.3f}")

            # LOGGING
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_param("n_estimators", 500)
            
            # Save the pipeline locally for the API
            os.makedirs("artifacts", exist_ok=True)
            with open("artifacts/model.pkl", "wb") as f:
                pickle.dump(pipeline, f)
            print("Saved pipeline to artifacts/model.pkl")

if __name__ == "__main__":
    trainer = ModelTrainer('data/processed/train_data.csv')
    trainer.run()