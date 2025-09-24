import sys
import os

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))

from main import load_and_preprocess_data, train_model

def test_data_loading():
    data = load_and_preprocess_data()
    assert not data.isnull().sum().any(), "Data contains missing values after preprocessing"


@pytest.mark.skip(reason="Intentional failure test for demo only")
def test_null_injection():
    data = load_and_preprocess_data()
    # artificially insert a null value into Age column
    data.loc[0, "Age"] = None

    # this should now fail because we forced a null
    assert not data.isnull().sum().any(), "Data contains missing values after preprocessing"


def test_model_accuracy():
    data = load_and_preprocess_data()
    model, accuracy = train_model(data)
    assert accuracy >= 0.75, "Model accuracy is below 75%"
