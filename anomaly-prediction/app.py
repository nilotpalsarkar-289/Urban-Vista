from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("test1.pkl")
    if not hasattr(model, "predict"):
        raise ValueError("Loaded object is not a valid model")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Ensure app doesn't crash if model loading fails

@app.route('/')
def home():
    return render_template('t.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        # Extract input values from form
        form_data = request.form
        input_features = [
            float(form_data.get("Day", 0)),                 # Day of the week
            float(form_data.get("Light", 0)),               # Light conditions
            float(form_data.get("Temperature", 0)),         # Sex of driver (renamed)
            float(form_data.get("Oxygen", 0)),              # Vehicle type (renamed)
            float(form_data.get("Humidity", 0)),            # Speed limit
            float(form_data.get("Road", 0)),                # Road type
            float(form_data.get("Passengers", 0)),          # Number of passengers
            float(form_data.get("Special", 0)),             # Special conditions (if any)
            float(form_data.get("Pedestrian_Crossing", 0))  # Pedestrian crossing presence
        ]

        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)  # Ensure 2D input

        # Perform prediction
        prediction = model.predict(input_array)[0]

        return render_template('t.html', pred=f"Predicted Severity Level: {prediction}")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Error processing prediction"}), 400

if __name__ == '__main__':
    app.run(debug=True)
