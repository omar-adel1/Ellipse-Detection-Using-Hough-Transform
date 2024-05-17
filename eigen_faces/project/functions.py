import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people,fetch_olivetti_faces
import matplotlib.pyplot as plt


def pca(X, k):
    # Standardize the data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std == 0, 1, X_std)  # avoid division by zero
    X_std = (X - X_mean) / X_std

    # Compute the covariance matrix
    cov_matrix = np.cov(X_std.T)  #cov = A^T * A

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)   

    # Sort the eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # Select the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]

    #   mapping back to origincal dataset
    X_pca = np.dot(X_std, top_k_eigenvectors)
    print(X_pca.shape)

    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:k] / np.sum(eigenvalues)

    return X_pca, explained_variance_ratio, top_k_eigenvectors, X_mean, X_std





def pca_transform(X, top_k_eigenvectors):
    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Project the data onto the new feature space
    X_pca = np.dot(X_std, top_k_eigenvectors)
    print(X_pca.shape)

    return X_pca

def pca_inverse_transform(X_pca, top_k_eigenvectors, X_mean, X_std):
    # Reconstruct the original data
    X_reconstructed = np.dot(X_pca, top_k_eigenvectors.T) * X_std + X_mean

    return X_reconstructed




def eigenFaces(num_components,variance,ds):

    if ds==1:
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        n_samples, h, w = lfw_people.images.shape
        X = lfw_people.data
        y = lfw_people.target
        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]
    else:
        olivetti_people = fetch_olivetti_faces()
        # Get the shape of the images
        n_samples, h, w = olivetti_people.images.shape
        # Get the data
        X = olivetti_people.data
        # Get the target
        y = olivetti_people.target
        # Get the target names
        target_names = np.array(["person_" + str(i) for i in range(40)])
        # Get the number of classes
        n_classes = target_names.shape[0]
   
   #split the data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Compute PCA (eigenfaces) on the face dataset 
    X_train_pca, explained_variance_ratio, top_k_eigenvectors, X_mean, X_std = pca(X_train, num_components)
    X_test_pca= pca(X_test, num_components)
    
    
    # Calculate the cumulative sum of explained variances
    explained_variances = np.cumsum(explained_variance_ratio)
    
    n_components = [np.argmax(explained_variances >= variance) + 1 ]
    
    images_reconstructed = []
    for n in n_components:
        X_train_pca = pca_transform(X_train, top_k_eigenvectors[:, :n])
        X_reconstructed = pca_inverse_transform(X_train_pca, top_k_eigenvectors[:, :n], X_mean, X_std)
        images_reconstructed.append(X_reconstructed)
        
        
    
    
    
    
    
    