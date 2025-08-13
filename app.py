from flask import Flask, render_template, request
import pickle
import numpy as np

# Load saved files
scaler = pickle.load(open('scaler.pkl', 'rb'))
dbscan = pickle.load(open('dbscan.pkl', 'rb'))
nn = pickle.load(open('nn.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        score = float(request.form['score'])

        # Scale the input
        scaled_data = scaler.transform([[income, score]])

        # Find nearest training point
        _, index = nn.kneighbors(scaled_data)
        cluster = labels[index[0][0]]

        return render_template('index.html', prediction_text=f"This customer belongs to cluster {cluster}")
    except:
        return render_template('index.html', prediction_text="Error in input. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
