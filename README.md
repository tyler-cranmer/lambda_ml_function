# task-ensemble-app

Task Enable App is an AWS Lambda function designed to predict the Action, Clarification, and Complexity of a developer task based on the input text data. It uses three NLP classification models to generate these predictions, which are then fed to a regression model to estimate the time for completion

## Features

- Takes developer task text data as input
- Uses three NLP classification models to predict:
  - Action: The type of action required to perform the task
  - Clarification: The level of clarity of the task description
  - Complexity: The complexity level of the task
- Feeds the predicted classifications to a regression model to estimate the time for completion

## Requirements

- Python 3.9
- AWS Lambda
- AWS API Gateway (optional, if you want to invoke the function using an API)
- SAM CLI

## Dependencies

- run the requirements.txt file
