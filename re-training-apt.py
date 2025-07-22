from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import psycopg2
import joblib
import lightgbm as lgb
from io import BytesIO
import requests
import os

app = FastAPI()

# Optional: allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Supabase DB config ----
SUPABASE_DB_USER = "postgres"
SUPABASE_DB_PASSWORD = "Hddua2004@"
SUPABASE_DB_NAME = "postgres"
SUPABASE_DB_PORT = "5432"
SUPABASE_DB_HOST = "db.qpofvsdudjiauslixazu.supabase.co"

SUPABASE_DB_URL = f"postgresql://{SUPABASE_DB_USER}:{SUPABASE_DB_PASSWORD}@{SUPABASE_DB_HOST}:{SUPABASE_DB_PORT}/{SUPABASE_DB_NAME}"

# ---- Storage URL ----
STORAGE_TRAIN_URL = "https://qpofvsdudjiauslixazu.supabase.co/storage/v1/object/public/apt-training/new_data.csv"

@app.get("/")
def root():
    return {"message": "Retraining API is running."}

@app.post("/retrain")
def retrain_model():
    try:
        # Step 1: Load original training dataset
        response = requests.get(STORAGE_TRAIN_URL)
        if response.status_code != 200:
            return {"status": "error", "message": f"Failed to download training data: {response.status_code}"}
        
        train_df = pd.read_csv(BytesIO(response.content))

        # Step 2: Connect to Supabase and fetch new flagged data
        conn = psycopg2.connect(SUPABASE_DB_URL)
        query = "SELECT * FROM flagged_apt"
        flagged_df = pd.read_sql_query(query, conn)
        conn.close()

        # Step 3: Combine datasets
        combined_df = pd.concat([train_df, flagged_df], ignore_index=True)

        if 'label' not in combined_df.columns:
            return {"status": "error", "message": "'label' column missing in combined dataset"}

        # Step 4: Split features and labels
        X = combined_df.drop(columns=["label"])
        y = combined_df["label"]

        # Step 5: Retrain model
        model = lgb.LGBMClassifier()
        model.fit(X, y)

        # Step 6: Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        return {"status": "success", "message": "Model retrained and saved to models/model.pkl."}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
