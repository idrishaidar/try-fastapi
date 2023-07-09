import json
from typing import Dict

from flask import Flask, request
from werkzeug.exceptions import HTTPException

from app import predict

app  = Flask(__name__)

@app.route("/", methods=["GET"])
async def index() -> Dict:
    return {
        "statusCode": 200,
        "body": "hi",
    }


@app.route("/predict_price", methods=["POST"])
def predict_price() -> Dict:
    if request.method == "POST":
        data = request.get_json()

        event = json.dumps({"body": data})

        pred = predict(event)
        status_code = pred.get("statusCode")

        if status_code != 200:
            raise HTTPException(status_code, detail=pred.get("body"))
        
        return pred
