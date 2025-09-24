from flask import Flask, request, jsonify
from main import load_and_preprocess_data, train_model

app = Flask(__name__)

# Load data and train the model
data = load_and_preprocess_data()
model, _ = train_model(data)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json

    # Ensure input data is provided
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    # Convert input data to a DataFrame (expected format for the model)
    import pandas as pd
    df = pd.DataFrame([input_data])

    # Make a prediction
    prediction = model.predict(df)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
