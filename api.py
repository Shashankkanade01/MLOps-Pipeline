from fastapi import FastAPI
# TODO: Load model from MLflow
# TODO: Define /predict endpoint

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}
