import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from PIL import Image

# https://youtu.be/SaEmG4wcFfg?si=DA3WVz-m5vhQSWuy



lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_train_stacked = np.column_stack((X_train_normalized))
X_train_stacked.shape #N^2 * M


cov_matrix = np.cov(X_train_stacked) #cov= A*A^T 
print(cov_matrix)

cov_matrix.shape
#### N^2 * N^2


cov_matrix2= np.cov(np.transpose(X_train_stacked))
cov_matrix2.shape

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix2)
# Sort the eigenvalues in descending order and get the indices

idx = np.argsort(eigenvalues)[::-1]

# Select the top k eigenvalues and their corresponding eigenvectors
k = 100  # replace with your desired number of components
top_k_eigenvalues = eigenvalues[idx[:k]]
top_k_eigenvectors = eigenvectors[:, idx[:k]]

# Transform the original dataset by projecting it onto the top k eigenfaces
X_train_transformed = np.dot(X_train_stacked, top_k_eigenvectors)


eigenvalues, eigenvectors = np.linalg.eig(cov_matrix2)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
####projecting the original dataset onto the top k eigenfaces
eigenconverted= np.dot(X_train_stacked, eigenvectors)

eigenfaces = eigenconverted.T/ np.sqrt((eigenconverted.T ** 2).sum(axis=1, keepdims=True))
eigenfaces.shape


_, axs = plt.subplots(2, 3, figsize=(10, 10))
axs = axs.flatten()
for i, (img, ax) in enumerate(zip(eigenfaces, axs)):
    ax.set_title(f"Eigenvalue: {np.round(eigenvalues[i], 2)}")
    ax.imshow(img.reshape(h, w), cmap="gray")
plt.show()


X_train_transformed = np.dot(X_train_stacked, top_k_eigenvectors)

# Normalize the test data and project it onto the same Eigenfaces
X_test_normalized = scaler.transform(X_test)
X_test_transformed = np.dot(X_test_normalized, top_k_eigenvectors)

# Train a classifier (like SVM) on the transformed training data
clf = SVC(C=1000, gamma=0.001)
clf.fit(X_train_transformed, y_train)

# Use the trained classifier to predict the labels of the transformed test data
y_pred = clf.predict(X_test_transformed)

# Evaluate the performance of the classifier
print(classification_report(y_test, y_pred, target_names=target_names))