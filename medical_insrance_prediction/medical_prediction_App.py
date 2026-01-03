from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
# Using a relative path assuming app.py is in the same directory as the model
model_path = 'medical_insurance_rf_model.pkl'
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Please ensure it is in the same directory.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction_text="Error: Model not loaded.")

    try:
        # Extract features from form
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Create DataFrame for model input
        # Note: Columns must match the training data feature names
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

        # Predict
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated Insurance Cost: ${output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in prediction: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
