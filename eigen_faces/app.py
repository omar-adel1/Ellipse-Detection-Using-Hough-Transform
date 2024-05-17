from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people,fetch_olivetti_faces
import matplotlib.pyplot as plt
from yarab import pca, pca_transform, pca_inverse_transform
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('UI.html')  # Replace 'UI.html' with the name of your HTML file

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    X = np.array(data['X'])
    y = np.array(data['y'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_pca, explained_variance_ratio, top_k_eigenvectors, X_mean, X_std = pca(X_train, 150)
    
    model = SVC()  # Create a new SVM model
    model.fit(X_train_pca, y_train)  # Train the model
    joblib.dump(model, 'model.pkl')  # Save the model to a file
    
    return jsonify({'message': 'Model trained successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array(data['X'])
    model = joblib.load('model.pkl')  # Load the model from the file
    prediction = model.predict(X)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)