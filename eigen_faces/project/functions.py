import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_lfw_people,fetch_olivetti_faces
import matplotlib.pyplot as plt
import warnings
import os


# Ignore matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def pca(X, k,variance):
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



    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:k] / np.sum(eigenvalues) 
    #The explained variance ratio is calculated for each principal component by dividing its eigenvalue by the sum of all eigenvalues.
    #This ratio represents the proportion of the dataset's variance that lies along the axis of each principal component.
    
    # Calculate the cumulative explained variance ratio
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    #The cumulative explained variance ratio is the cumulative sum of the explained variance ratios.
    # It represents the total variance explained by the first n components. For example, cumulative_explained_variance_ratio[i] is the total variance explained by the first i+1 components.

    # Find the smallest k such that the cumulative explained variance ratio is at least v%
    v = np.where(cumulative_explained_variance_ratio >= float(variance))[0][0] + 1
    print(v)
    
    #Find the smallest k such that the cumulative explained variance ratio is at least variance: 
    # This step finds the smallest number of components that explain at least a variance proportion of the total variance.
    # The np.where function returns the indices where cumulative_explained_variance_ratio >= float(variance) is true, and [0][0] + 1 selects the first such index and adds 1 to it (because indices are 0-based).
    selected_components = eigenvectors[:, :v]
    
    # projection
    X_pca = np.dot(X_std, selected_components)
    top_k_eigenvectors=selected_components
    

    

    

    return X_pca, top_k_eigenvectors, X_mean, X_std





def pca_inverse_transform(X_pca, top_k_eigenvectors, X_mean, X_std):
    # Map back to the original space
    X_std_reconstructed = np.dot(X_pca, top_k_eigenvectors.T)

    # Unstandardize the data
    X_reconstructed = X_std_reconstructed * X_std + X_mean

    return X_reconstructed




def eigenFaces(num_components,variance,ds):

    if ds==1:
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        n_samples, h, w = lfw_people.images.shape
        X = lfw_people.data
        y = lfw_people.target
    else:
        olivetti_people = fetch_olivetti_faces()
        # Get the shape of the images
        n_samples, h, w = olivetti_people.images.shape
        # Get the data
        X = olivetti_people.data
        # Get the target
        y = olivetti_people.target
   
   
   

    
   #split the data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    
    # Compute PCA (eigenfaces) on the face dataset 
    X_train_pca, top_k_eigenvectors, X_mean, X_std = pca(X_train, num_components,variance)
  
    
    # Select the top k eigenfaces, project the training set onto the eigenfaces, and reconstruct the images
    X_reconstructed = pca_inverse_transform(X_train_pca, top_k_eigenvectors, X_mean, X_std)
    
    
    # Select the first 10 images from the array
    # Get the first 16 original and reconstructed images
    first_16_original_images = X_train[:16]
    first_16_reconstructed_images = X_reconstructed[:16]

    # Create a subplot with 8 rows and 4 columns (2 sets of 4x4 images)
    fig, axes = plt.subplots(8, 4, figsize=(10, 15),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i in range(16):
        # Reshape the original and reconstructed images to their original 2D shape
        original_image_2D = first_16_original_images[i].reshape((h, w))
        reconstructed_image_2D = first_16_reconstructed_images[i].reshape((h, w))

        # Display the original image in the top 4 rows and the reconstructed image in the bottom 4 rows
        axes[i // 4, i % 4].imshow(original_image_2D.real, cmap='gray')
        axes[i // 4 + 4, i % 4].imshow(reconstructed_image_2D.real, cmap='gray')
        
       

# Save the figure

    
    file_path = '..\\image\\Ellipse-Detection-Using-Hough-Transform\\eigen_faces\\project\\generated_image.png'

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, delete it
        os.remove(file_path)


    # Save the figure as an image in the same directory
    plt.savefig('..\\image\\Ellipse-Detection-Using-Hough-Transform\\eigen_faces\\project\\generated_image.png')
    
   
        
        
    
    
    
    
    
    