import json
import os

import numpy as np

from joblib import load

import preprocess

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

import xgboost as xgb


# Action ML model
current_directory = os.getcwd()
action_model_file = current_directory + '/models/action_estimator.joblib'
action_model = load(action_model_file)

# Clarifiation ML model
clarify_model_file = current_directory + '/models/clarify_estimator.joblib'
clarify_model = load(clarify_model_file)

# Complexity ML model
complexity_model_file = current_directory + \
    '/models/complexity_estimator.joblib'
complexity_model = load(complexity_model_file)

# Time ML model
time_model_file = current_directory + '/models/time_estimator.xgb'
model = xgb.Booster()
model.load_model(time_model_file)


def time_estimate(complexity: str, action: str, clarification: str) -> int:

    actions = ['add new', 'remove', 'troubleshoot', 'update']
    clarifications = ['content', 'custom post type', 'feature', 'form',
                      'google analytics', 'other', 'page', 'plugin', 'product', 'script',
                      'section', 'speed optimization', 'style', 'website']
    complexity = int(complexity)

    if complexity > 5 or complexity < 1:
        raise ValueError('Invalid input for complexity. Must be between 1 - 5')
    if action not in actions:
        raise ValueError(
            f"Invalid input for action. Must be one of these options: {actions}")
    if clarification not in clarifications:
        raise ValueError(
            f"Invalid input for clarification. Must be one of these options: {clarification}")

    result = np.zeros((1, len(actions + clarifications) + 1), dtype=int)
    result[0, 0] = complexity

    return result


def lambda_handler(event, content):
    # get query string param inputs
    client_request = event['queryStringParameters']['request']

    print(f"Client Request: {client_request}")

    text = preprocess.clean_text(client_request)

    action = action_model.predict([text])
    clarification = clarify_model.predict([text])
    complexity = complexity_model.predict([text])

    prediction = model.predict(xgb.DMatrix(
        time_estimate(complexity[0], action[0], clarification[0])))

    res_body = {}
    res_body['complexity'] = complexity[0]
    res_body['action'] = action[0]
    res_body['clarification'] = clarification[0]
    res_body['predicted_time'] = round(prediction[0])

    http_res = {}
    http_res['isBase64Encoded'] = 'false'
    http_res['statusCode'] = 200
    http_res['headers'] = {
        "Access-Control-Allow-Origin": "https://dash.moreseconds.com/",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Content-Type"
    }
    http_res['headers']['Content-type'] = 'application/json'
    http_res['body'] = json.dumps(res_body)

    return http_res
