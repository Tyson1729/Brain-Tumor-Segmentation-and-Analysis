import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Stryden | Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.header("Team Details")
    st.markdown("""
    **Team ID:** 22CDP11
    
    **Members:**
    - Sheewam Kumar
    - Ayush Anand
    - Yashvi Bhuwalka
    - Dia Sharma
    """)
    st.info("Please log in via the 'User's Credentials' page to begin.")

# --- Main Page ---
st.title("Stryden: A Comprehensive Brain Tumor Analysis Suite")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    # Using a placeholder for image URL
    st.image("https://mdwestone.com/wp-content/uploads/2024/07/brain-tumor.jpg", caption="AI-Powered Medical Imaging Analysis")

with col2:
    st.markdown("""
    ### Welcome to Stryden!

    This project leverages state-of-the-art deep learning models to assist in the analysis of brain MRI scans. Our goal is to provide a powerful, user-friendly platform for both educational and research purposes.

    **To get started, please navigate to the `User's Credentials` page to log in.** Your role (Patient, Doctor, or Institute) will determine your access to the tools.

    **Our application offers four core functionalities:**

    1.  **3D Tumor Segmentation (Doctor/Institute Access):**
        - Upload multi-sequence 3D NIfTI files (`T1CE`, `T2`, `FLAIR`).
        - Our 3D U-Net model will process the entire volume and generate a segmentation mask, highlighting the precise location and structure of the tumor.

    2.  **2D Tumor Segmentation (Doctor/Institute Access):**
        - Upload a single 2D MRI image (`JPEG`, `PNG`).
        - Our new 2D U-Net model will analyze the slice and generate a precise segmentation mask of the tumor region.

    3.  **Brain Tumor Classification (All Logged-in Users):**
        - Upload a 2D MRI image (`JPEG`, `PNG`).
        - A Convolutional Neural Network (CNN) will classify the image into one of four categories: **Glioma, Meningioma, Pituitary tumor, or No Tumor**.

    4.  **Stryden.ai Assistant (All Logged-in Users):**
        - An intelligent chatbot powered by Google's Gemini model.
        - It provides context-aware information. If you classify a tumor, you can ask specific questions about that tumor type. It can also answer general questions about brain tumors and the application's functionality.
    
    5.  **Session Dashboard (Doctor/Institute Access):**
        - Review all user data, classification results, and segmentation images from the current session in one place.

    """, unsafe_allow_html=True)

st.markdown("---")
st.success("This tool is for informational and educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.")