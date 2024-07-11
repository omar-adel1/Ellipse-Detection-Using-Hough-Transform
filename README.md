# Image Processing Projects

This repository contains two projects focused on image processing: Ellipse Detection Using Hough Transform and Eigen Faces. Both projects demonstrate different techniques and algorithms for processing and analyzing images.

## Table of Contents

1. Ellipse Detection Using Hough Transform
   - **Project Structure**
     - app.py: The main application script that sets up the Streamlit interface and handles user inputs.
     - EllipseDetectionHoughFunctions.py: Contains the core functions for loading and preprocessing images and performing ellipse detection using the Hough Transform.
   - **Getting Started**
     - **Prerequisites**: Python 3.7+, Streamlit, OpenCV, NumPy, PIL (Pillow)
     - **Installation**:
       - Clone the repository:
         ```bash
         git clone https://github.com/yourusername/Ellipse-Detection-Using-Hough-Transform.git
         cd Ellipse-Detection-Using-Hough-Transform
         ```
       - Create a virtual environment (optional but recommended):
         ```bash
         python -m venv venv
         source venv/bin/activate  # On Windows, use venv\Scripts\activate
         ```
       - Install the required packages:
         ```bash
         pip install -r requirements.txt
         ```
     - **Running the Application**:
       ```bash
       streamlit run app.py
       ```
   - **Usage**:
     - Open your web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).
     - Upload an image using the file uploader in the sidebar.
     - Adjust the parameters for ellipse detection using the sliders.
     - Click the "Detect Ellipses" button to perform ellipse detection.
     - The original and processed images will be displayed side by side.
   - **Files and Functions**:
     - app.py: Sets up the Streamlit interface, handles file upload and user input, and displays images.
     - EllipseDetectionHoughFunctions.py: Functions for image preprocessing and ellipse detection using the Hough Transform.

2. Eigen Faces
   - **Project Structure**
     - server.py: The main server script that sets up the Flask application and handles requests.
     - functions.py: Contains the core functions for PCA computation and image reconstruction.
     - templates/main.html: HTML template for the user interface.
   - **Getting Started**
     - **Prerequisites**: Python 3.7+, Flask, scikit-learn, NumPy, Matplotlib
     - **Installation**:
       - Clone the repository:
         ```bash
         git clone https://github.com/yourusername/Eigen-Faces.git
         cd Eigen-Faces
         ```
       - Create a virtual environment (optional but recommended):
         ```bash
         python -m venv venv
         source venv/bin/activate  # On Windows, use venv\Scripts\activate
         ```
       - Install the required packages:
         ```bash
         pip install -r requirements.txt
         ```
     - **Running the Application**:
       ```bash
       python server.py
       ```
   - **Usage**:
     - Open your web browser and navigate to the URL provided by Flask (typically `http://localhost:5000`).
     - Upload an image using the file uploader in the interface.
     - Set the number of components and variance threshold using the provided inputs.
     - Select the dataset (either LFW or Olivetti Faces).
     - Click the "Process" button to perform PCA and generate Eigenfaces.
     - The reconstructed images will be displayed.
   - **Files and Functions**:
     - server.py: Sets up the Flask application, defines routes, and handles image processing logic.
     - functions.py: Functions for PCA computation, image reconstruction, and image saving.

---

## Example Usage

The projects provide examples of implementing image processing techniques:
- Ellipse Detection Using Hough Transform showcases image preprocessing and feature detection.
- Eigen Faces demonstrates PCA for feature extraction and image reconstruction in face recognition.

Both projects offer interactive interfaces for users to upload images, adjust parameters, and visualize results directly in their browsers.

---

| Name | GitHub | LinkedIn | Project |
| ---- | ------ | -------- | -------- |
| Omar Adel Hassan | [@Omar_Adel](https://github.com/omar-adel1) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omar-adel-59b707231/) | Ellipse Detection Using Hough Transform |
| Sharif Ehab | [@Sharif_Ehab](https://github.com/SharifEhab) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sharif-elmasry-b167a3252/) | Ellipse Detection Using Hough Transform |
| Mostafa Khaled | [@Mostafa_Khaled](https://github.com/MostafaDarwish93) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mostafa-darwish-75a29225b/) | Eigen Faces |
| Bahey Ismail | [@Bahey_Ismail ](https://github.com/Bahey1200022) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bahey-ismail-1602431a4/) | Eigen Faces |
