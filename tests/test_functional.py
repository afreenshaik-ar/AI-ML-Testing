import pytest
# from main import load_and_preprocess_data, train_model
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_and_preprocess_data, train_model


@pytest.fixture
def model():
    # Load data and train the model
    data = load_and_preprocess_data()
    model, _ = train_model(data)  # Unpack model and ignore accuracy
    return model

def test_valid_input(model):
    # Match the training features
    input_data = pd.DataFrame([{
    "Pclass": 1,       # 1st class
    "Age": 29,         # age in years
    "SibSp": 0,        # number of siblings/spouses aboard
    "Parch": 0,        # number of parents/children aboard
    "Fare": 100.0,     # ticket fare
    "Sex_male": 1,     # 1 = male, 0 = female
    #"Embarked_Q": 0,   # 1 if embarked at Queenstown
    "Embarked_S": 1    # 1 if embarked at Southampton
    }])
    prediction = model.predict(input_data)
    assert prediction in [[0], [1]], "Prediction should be 0 or 1"

@pytest.mark.negative()
def test_invalid_input(model):
    # Invalid input data (missing columns)
    input_data = pd.DataFrame([{
        "Pclass": 1,
        "Sex_male": 0,
        "Age": 29
    }])
    with pytest.raises(Exception):
        model.predict(input_data)

def test_edge_case_input(model):
    # Extreme passenger input
    input_data = pd.DataFrame([{
        "Pclass": 3,
        "Age": 1,        # Minimum age
        "SibSp": 8,      # Max siblings/spouse
        "Parch": 8,      # Max parents/children
        "Fare": 512.3,   # High fare
        "Sex_male": 0,   # One-hot encoded
        "Embarked_Q": 0, # May or may not exist in training
        "Embarked_S": 1  # One-hot encoded
    }])

    # Align with model’s expected features
    expected_features = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(input_data)
    assert prediction in [[0], [1]], "Prediction should be 0 or 1"
