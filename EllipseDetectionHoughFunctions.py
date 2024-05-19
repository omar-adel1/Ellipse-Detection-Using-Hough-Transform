import numpy as np
import cv2
from PIL import Image

def load_and_preprocess_image(image_file):
    """
    This function loads an image from a file, converts it to a grayscale image,
    applies a Gaussian blur, and then performs edge detection using the Canny algorithm.
    It returns the original image as a numpy array, the grayscale image, and the edges detected.

    Args:
    - image_file: A file object or a path to an image file.

    Returns:
    - image_np: The original image in numpy array format.
    - gray: The grayscale version of the image.
    - edges: The edges detected in the image using the Canny algorithm.
    """
    # Convert the image file to numpy array
    image = Image.open(image_file)
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_edge_threshold = 0.2 * np.max(blurred)
    max_edge_threshold = 0.6 * np.max(blurred)
    
    # Apply Canny edge detector
    edges = cv2.Canny(gray, min_edge_threshold, max_edge_threshold, L2gradient=True)
    return image_np, gray, edges

def hough_transform_ellipse_detection(image, edges, min_major_axis, max_major_axis, min_minor_axis, max_minor_axis, threshold, num_edge_points):
    """
    This function detects ellipses in an image using the Hough Transform method.

    Args:
    - image: The original image in numpy array format.
    - edges: The edges detected in the image.
    - min_major_axis: Minimum length of the major axis of potential ellipses.
    - max_major_axis: Maximum length of the major axis of potential ellipses.
    - min_minor_axis: Minimum length of the minor axis of potential ellipses.
    - max_minor_axis: Maximum length of the minor axis of potential ellipses.
    - threshold: The minimum proportion of votes needed to consider an ellipse as detected.
    - num_edge_points: The number of edge points to sample for ellipse detection.

    Returns:
    - result_image: The original image with detected ellipses drawn on it.
    - accumulator: The accumulator dictionary with votes for potential ellipses.
    """
    
    # Identify the coordinates of all edge points in the image
    edge_points_indices = np.argwhere(edges)
    # Randomly shuffle the edge points
    np.random.shuffle(edge_points_indices)
    # Select a subset of edge points for processing
    edge_points_indices = edge_points_indices[:num_edge_points]

    # Precompute trigonometric values (cosine and sine) for angles 0 to 179 degrees
    cos_theta = np.cos(np.deg2rad(np.arange(180)))
    sin_theta = np.sin(np.deg2rad(np.arange(180)))

    # Initialize an empty dictionary to act as an accumulator for votes
    accumulator = {}

    # Iterate through each sampled edge point
    for y, x in edge_points_indices:
        # Iterate through each possible major axis length
        for A in range(min_major_axis, max_major_axis):
            # Iterate through each possible minor axis length
            for B in range(min_minor_axis, max_minor_axis):
                # Iterate through each angle theta from 0 to 179 degrees
                for theta in range(180):
                    # Calculate the potential center coordinates of the ellipse
                    x_ellipse_center = int(round(x - A * cos_theta[theta]))
                    y_ellipse_center = int(round(y - B * sin_theta[theta]))
                    
                    # Ensure the calculated center is within the image bounds
                    if 0 <= x_ellipse_center < image.shape[1] and 0 <= y_ellipse_center < image.shape[0]:
                        # Create a key representing the ellipse parameters
                        dict_key = (y_ellipse_center, x_ellipse_center, A, B, theta, phi)
                        # Increment the vote count for this key in the accumulator
                        if dict_key in accumulator:
                            accumulator[dict_key] += 1
                        else:
                            accumulator[dict_key] = 1  

    # Calculate the threshold value for detecting valid ellipses
    threshold_value = max(accumulator.values()) * threshold
    # Identify potential ellipses that have votes exceeding the threshold
    potential_ellipses = [dict_key for dict_key, value in accumulator.items() if value >= threshold_value]
    print(f"Potential ellipses identified: {len(potential_ellipses)}")

    # Create a copy of the original image for drawing detected ellipses
    result_image = np.copy(image)
    scale_factor = 0.1  # Scale down the size of the drawn ellipses for visualization

    # Draw each detected ellipse on the result image
    for y_ellipse_center, x_ellipse_center, A, B, theta, phi in potential_ellipses:
        # Scale down the major and minor axes lengths
        scaled_a = int(A * scale_factor)
        scaled_b = int(B * scale_factor)
        # Draw the ellipse on the image
        cv2.ellipse(result_image, (x_ellipse_center, y_ellipse_center), (scaled_a, scaled_b), np.degrees(theta), 0, 360, (0, 255, 0), 1)
    print("Ellipses drawn on image.")

    return result_image, accumulator
