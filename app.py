import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load the model, scaler, and SMOTE
with open('RF_model.pkl', 'rb') as file:
    model = pickle.load(file)  # Trained model

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)  # Scaler for preprocessing

# SMOTE is not used for prediction, so it is not loaded

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get the data from the API request
        data = request.json['data']
        print("Input Data:", data)

        # Convert data to a numpy array and reshape it
        new_data = np.array(list(data.values())).reshape(1, -1)

        # Scale the new data using the pre-trained scaler
        new_data_scaled = scaler.transform(new_data)

        # Make predictions using the trained model
        output = model.predict(new_data_scaled)
        print("Model Output:", output[0])

        # Return the prediction as JSON
        return jsonify({'prediction': int(output[0])})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form submission
        data = [float(x) for x in request.form.values()]

        # Reshape the data and scale it using the pre-trained scaler
        final_input = np.array(data).reshape(1, -1)
        final_input_scaled = scaler.transform(final_input)
        print("Scaled Input:", final_input_scaled)

        # Make predictions using the trained model
        output = model.predict(final_input_scaled)[0]

        # Interpret the prediction (assuming binary classification for "Good" or "Bad")
        prediction_text = "Good Quality" if output == 1 else "Bad Quality"

        # Render the prediction result on the HTML page
        return render_template("home.html", prediction_text=f"The predicted quality is: {prediction_text}")

    except Exception as e:
        print(f"Error: {e}")
        return render_template("home.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
