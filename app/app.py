import json
import os

import numpy as np

from joblib import load

from preprocess import clean_text, generate_features, generate_complexity_features

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Action ML model
current_directory = os.getcwd()
action_model_file = current_directory + '/models/action_estimator_0418.joblib'
action_model = load(action_model_file)

# Clarifiation ML model
clarify_model_file = current_directory + '/models/clarification_estimator_0418.joblib'
clarify_model = load(clarify_model_file)

# Complexity ML model
complexity_model_file = current_directory + \
    '/models/complexity_estimator_0418.joblib'
complexity_model = load(complexity_model_file)

# Time ML model
time_model_file = current_directory + '/models/time_estimator_0418.joblib'
time_model = load(time_model_file)


def lambda_handler(event, content):
    # get query string param inputs
    client_request = event['queryStringParameters']['request']

    print(f"Client Request: {client_request}")

    text = clean_text(client_request)

    action = action_model.predict([text])
    clarification = clarify_model.predict([text])
    complexity = complexity_model.predict(generate_complexity_features(action[0], clarification[0]))
    prediction = time_model.predict(generate_features(
        complexity[0], action[0], clarification[0]))

    res_body = {}
    res_body['complexity'] = complexity[0]
    res_body['action'] = action[0]
    res_body['clarification'] = clarification[0]
    res_body['predicted_time'] = round(np.expm1(prediction[0]))

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
