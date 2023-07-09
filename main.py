import json
from typing import Dict

from fastapi import FastAPI, HTTPException, Request

from app import predict

app  = FastAPI()

@app.get("/")
async def index() -> Dict:
    return {
        "statusCode": 200,
        "body": "hi",
    }


@app.post("/predict_price")
async def predict_price(request: Request) -> Dict:
    body = await request.body()
    body = json.loads(body)
    event = json.dumps({"body": body})

    pred = predict(event)
    status_code = pred.get("statusCode")

    if status_code != 200:
        raise HTTPException(status_code, detail=pred.get("body"))
    
    return pred
