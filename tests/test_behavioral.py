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


@pytest.mark.behavioral()
def test_prediction_consistency(model):
    input_data = pd.DataFrame([{
        "Pclass": 2, "Age": 30, "SibSp": 1, "Parch": 0,
        "Fare": 50, "Sex_male": 1, "Embarked_S": 1
    }])
    pred1 = model.predict(input_data)
    print("Prediction result: ", pred1)
    pred2 = model.predict(input_data)
    print("Prediction result: ", pred2)
    assert (pred1 == pred2).all(), "Model predictions are inconsistent"

@pytest.mark.fairness
@pytest.mark.xfail(reason="Known bias: model performance differs between male and female")
def test_gender_bias(model):
    data = load_and_preprocess_data()   # reload data here

    male = data[data['Sex_male'] == 1]
    female = data[data['Sex_male'] == 0]

    male_acc = model.score(male.drop("Survived", axis=1), male["Survived"])
    female_acc = model.score(female.drop("Survived", axis=1), female["Survived"])

    diff = abs(male_acc - female_acc)
    print(f"Male Accuracy: {male_acc:.2f}, Female Accuracy: {female_acc:.2f}, Diff: {diff:.2f}")

    assert diff < 0.15, f"Model biased! Male={male_acc:.2f}, Female={female_acc:.2f}"
