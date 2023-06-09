import json
import os
from joblib import load
from preprocess import Models


# Complexity Model
current_directory = os.getcwd()
complexity_model_file = current_directory + "/models/complexity_06_07.joblib"
complexity_model = load(complexity_model_file)

# Time Estimation Model
time_model_file = current_directory + "/models/time_est_06_07.joblib"
time_model = load(time_model_file)


def lambda_handler(event, content):
    # get query string param inputs
    dev_type = event["queryStringParameters"]["dev_type"]
    dev_cat = event["queryStringParameters"]["dev_category"]

    print(f"Dev Type: {dev_type}\nDev Category: {dev_cat}")

    classification = Models(complexity_model=complexity_model, time_model=time_model)
    complexity = classification.complexity_predict(
        dev_type=dev_type, dev_category=dev_cat
    )
    time_est = classification.time_predict(
        dev_type=dev_type, dev_category=dev_cat, complexity=complexity
    )

    res_body = {}
    res_body["complexity"] = complexity
    res_body["predicted_time"] = round(time_est)

    http_res = {}
    http_res["isBase64Encoded"] = "false"
    http_res["statusCode"] = 200
    http_res["headers"] = {
        "Access-Control-Allow-Origin": "https://dash.moreseconds.com/",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    http_res["headers"]["Content-type"] = "application/json"
    http_res["body"] = json.dumps(res_body)

    return http_res
