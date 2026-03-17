from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(final_features)

        # Convert to class name
        classes = ["Setosa", "Versicolor", "Virginica"]
        output = classes[prediction[0]]

        return render_template("index.html", prediction_text=f"Predicted: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error in prediction")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)