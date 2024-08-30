from flask import Flask, request, render_template
import pickle
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    form_data = request.form.to_dict()

    # Initialize list for feature values
    int_features = []

    # Map for referral source
    referral_source_map = {'other': 0, 'SVI': 1, 'SVHC': 2, 'STMW': 3, 'SVHD': 4}

    # Convert form data to feature values
    for key, value in form_data.items():
        try:
            if key in ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 
                       'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
                       'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
                       'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 
                       'TBG_measured', 'binaryClass']:
                int_features.append(int(value))
            elif key in ['TSH', 'T3', 'TT4', 'T4U', 'FTI']:
                int_features.append(float(value))
            elif key == 'referral_source':
                int_features.append(referral_source_map.get(value, -1))  # Handle unknown values
            else:
                int_features.append(int(value))  # Default case for other fields
        except ValueError:
            # Handle cases where conversion to int or float fails
            int_features.append(0)  # Use 0 or any default value for failed conversions

    # Log the features for debugging
    logging.debug("Features: %s", int_features)

    # Ensure the correct number of features
    expected_num_features = 32  # Update this to match your model's expected number of features
    if len(int_features) != expected_num_features:
        logging.error("Feature shape mismatch: Expected %d, got %d", expected_num_features, len(int_features))
        return "Error: Incorrect number of features. Expected {} but got {}.".format(expected_num_features, len(int_features)), 400

    final_features = [np.array(int_features)]

    # Make prediction
    try:
        prediction = model.predict(final_features)
        output = 'P' if prediction[0] == 1 else 'N'
    except Exception as e:
        logging.error("Prediction error: %s", str(e))
        return str(e), 500

    logging.info("Prediction result: %s", output)

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
