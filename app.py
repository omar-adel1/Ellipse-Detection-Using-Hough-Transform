import streamlit as st
from PIL import Image
import EllipseDetectionHoughFunctions as functions


st.set_page_config(page_title="Hough Transform", page_icon=":radio:", layout="wide")

# Sidebar file uploader
with st.sidebar:
    st.header("Parameters")
    min_major_axis = st.slider("Min Major Axis", 10, 100, 10)
    max_major_axis = st.slider("Max Major Axis", 10, 200, 20)
    min_minor_axis = st.slider("Min Minor Axis", 5, 50, 5)
    max_minor_axis = st.slider("Max Minor Axis", 5, 100, 10)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.1)
    num_edge_points = st.slider("Number of Edge Points", 10, 1000, 100)

st.header("Ellipse Detection Using Hough Transform")

# Display original image in the first column
col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if file is not None:
        image_np, gray, edges = functions.load_and_preprocess_image(file)
        st.image(Image.fromarray(edges), caption="Edges (Before Detection)", use_column_width=True)
    else:
        st.write("Please upload an image.")

# Sliders for ellipse ranges, threshold, and num_edge_points
with col2:
    st.text('After')
# Button to trigger ellipse detection
    if st.button("Detect Ellipses"):
        if file is not None:
            result_image, accumulator = functions.hough_transform_ellipse_detection(image_np,edges, min_major_axis, max_major_axis, min_minor_axis, max_minor_axis, threshold, num_edge_points)
            st.image(result_image, caption='After Hough Transform', use_column_width=True)
        else:
            st.warning("Please upload an image first.")