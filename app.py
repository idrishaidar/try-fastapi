import json
import pickle

import pandas as pd


model_filename = "final_model.sav"

# load the model
model = pickle.load(open(model_filename, "rb"))

def predict(event):
    if isinstance(event, str):
        event_dict = json.loads(event)
    else:
        event_dict = event

    body = event_dict.get("body", event_dict)

    X_pred = pd.DataFrame.from_dict([body])
    y_pred = model.predict(X_pred)[0]

    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": y_pred}),
        "headers": {"Content-Type": "application/json"},
    }
