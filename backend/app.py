from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import numpy as np
from pydantic import BaseModel


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="""An API that utilises a Machine Learning model that detects 
                if a credit card transaction is fraudulent or not based on the 
                following features: hours, amount, transaction type etc.""",
    version="1.0.0", debug=True)


templates = Jinja2Templates(directory="templates")


# @app.get("/", response_class=PlainTextResponse)
# async def running():
#     note = """
#             Credit Card Fraud Detection API 🙌🏻
#             Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
#           """
#     return note


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


favicon_path = 'favicon.png'


@app.get('/favicon.png', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


class FraudDetection(BaseModel):
    step: int
    types: int
    amount: float
    oldbalanceorig: float
    newbalanceorig: float
    oldbalancedest: float
    newbalancedest: float
    isflaggedfraud: float


@app.post('/predict')
def predict(data: FraudDetection):
    features = np.array([[data.step, data.types, data.amount, data.oldbalanceorig, data.newbalanceorig,
                          data.oldbalancedest, data.newbalancedest, data.isflaggedfraud]])

    with open('fraud_detection_model.pickle', 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(features)
    if predictions == 1:
        return {"fraudulent"}
    elif predictions == 0:
        return {"not fraudulent"}