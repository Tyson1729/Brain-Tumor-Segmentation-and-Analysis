import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from architecture import CNN2D # Import from architecture.py

# --- Page Configuration & Helpers ---
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")
st.title("🧠 Brain Tumor Classification")

def logout():
    """Clears all session data and logs the user out."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    st.experimental_rerun()

def check_access(allowed_roles):
    """Checks if the user is logged in."""
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in first from the 'User's Credentials' page.")
        return False
    return True

# Add logout button
st.markdown('<div style="position: absolute; top: 10px; right: 10px;"><a href="#" id="logout-button"></a></div>', unsafe_allow_html=True)
if st.button("Log Out"):
    logout()

# --- Access Control Check ---
if not check_access(allowed_roles=['patient', 'doctor', 'institute']):
    st.stop()
    
st.markdown("---")

with st.expander("ℹ️ How to Use This Page"):
    st.markdown("""
    This page uses a **Convolutional Neural Network (CNN)** to classify a 2D MRI brain scan image.

    **1. Upload an Image:**
        - Provide an image in **JPEG, JPG, or PNG** format.
    
    **2. Model Classification:**
        - The model will classify the image into:
            - **Glioma**
            - **Meningioma**
            - **Pituitary**
            - **No Tumor**

    **3. Ask the AI:**
        - After a successful classification, navigate to the **Stryden.ai** page. 
        - The assistant will have context about your result and can answer specific questions!
    """)

# --- Model Loading ---
@st.cache_resource
def load_classification_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN2D(num_classes=4).to(device) # Use imported class
    model_path = r"models/classification.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_classification_model()

# 3. Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet norms
])

# Class labels
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# 4. Streamlit UI
st.markdown("""
Upload an MRI brain scan image (**JPEG/JPG/PNG**) to detect the type of tumor.  
⚠️ Please upload only MRI brain scan images.
""")

uploaded_file = st.file_uploader("📤 Upload an MRI scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("Classifying..."):
                img_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_class = class_names[predicted_idx.item()]
                confidence_score = confidence.item()

                if confidence_score < 0.60:
                    st.error("❌ Irrelevant context detected or low confidence. Please upload a valid brain MRI image.")
                    # Clear any previous context
                    if 'tumor_class' in st.session_state:
                        del st.session_state['tumor_class']
                else:
                    st.success(f"✅ Predicted Tumor Type: **{predicted_class.replace('_', ' ').title()}**")
                    st.info(f"Confidence Score: **{confidence_score*100:.2f}%**")
                    
                    # Store the result in session state for Stryden.ai
                    st.session_state.tumor_class = predicted_class
                    
                    if 'user_data' in st.session_state:
                        st.session_state.session_log.append({"user": st.session_state.user_data.get("Name"), "action": "Classification", "file": uploaded_file.name, "result": predicted_class})
                    
                    st.markdown("---")
                    st.markdown("👉 You can now go to the **Stryden.ai** page to ask questions about this result!")

    except Exception as e:
        st.error(f"⚠️ Error processing the image. Please upload a valid MRI scan. Error: {e}")
