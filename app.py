# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Load persisted scaler and model (Python 3.12 compatible)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/assessment')
def assessment_form():
	return render_template('main.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Cast all inputs to numeric types expected by the model
        age = int(request.form['age'])
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form.get('slope'))
        ca = int(request.form['ca'])
        thal = int(request.form.get('thal'))

        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)
        prediction = int(pred[0])

        return render_template('result.html', prediction=prediction)
        
        

if __name__ == '__main__':
	# In production, a WSGI server (e.g., gunicorn) will run `app:app`.
	# This block is only for local/dev runs.
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)

