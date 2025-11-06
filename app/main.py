from importlib import reload
from math import factorial
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from app.load_model import LoadModel

class InputData(BaseModel):
    text:str

def create_app() -> FastAPI:
    app = FastAPI(name="Hate Speech Detection API", 
                description="API for hate speech detection",
                version="1.0.0")

    @app.on_event("startup")
    def load_model():
        global model
        model = LoadModel()

    @app.post("/predict")
    async def predict(input_data: InputData):
        result = model.predict(input_data.text)
        return {"result": result}

    
    return app

if __name__ == "__main__":
    uvicorn.run("app.main:create_app", host="localhost", port=8000, reload=True)