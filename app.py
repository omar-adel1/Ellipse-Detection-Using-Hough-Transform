import streamlit as st 
from PIL import Image

st.set_page_config(page_title="Hough Transform", page_icon=":radio:", layout="wide")
with st.sidebar:
    file = st.file_uploader("Choose an image", type=["jpg", "jfif","png"], accept_multiple_files=False)

st.header("Ellipse Detection Using Hough Transform")

col1, col2 = st.columns(2)
with col1:
    if file is not None:
        image = Image.open(file)
        resized_image = image.resize((190, 190))
        st.image(resized_image, caption='Before Hough Transform', use_column_width=True)
    else:
        st.write("Please upload an image.")
with col2:
    st.text('After')
