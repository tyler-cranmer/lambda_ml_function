# task-ensemble-app

Task Ensemble App is an AWS Lambda function designed to predict the complexity of a task and then its fed into a regression model to estimate the time for completion.

## Features

- Takes developer task text data as input
- Uses 2 ML models to predict:
  - Complexity: The complexity level of the task
  - Predicted Time for Task: The amount of time a task should take to complete.
- Feeds the predicted classifications to a regression model to estimate the time for completion

## Requirements

- Python 3.9
- AWS Lambda
- AWS API Gateway (optional, if you want to invoke the function using an API)
- SAM CLI
- DOCKER

## Dependencies

- run the requirements.txt file
