import os
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st
import torch
import random
from architecture import UNet3D, UNet2D # Import both models from architecture.py
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io

# --- Page Configuration & Helpers ---
st.set_page_config(page_title="Tumor Segmentation", layout="wide")
st.title("🧠 Brain Tumor Segmentation")

def logout():
    """Clears all session data and logs the user out."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    # st.experimental_rerun()

def check_access(allowed_roles):
    """Checks if the user is logged in and has the correct role."""
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in first from the 'User's Credentials' page.")
        return False
    
    role = st.session_state.get('role', 'patient')
    if role not in allowed_roles:
        st.error("Access Denied: Your user role does not have permission to view this page.")
        st.error("Only 'Doctor' or 'Institute' roles can access Segmentation tools.")
        return False
    return True

# Add logout button to top right
st.markdown('<div style="position: absolute; top: 10px; right: 10px;"><a href="#" id="logout-button"></a></div>', unsafe_allow_html=True)
if st.button("Log Out"):
    logout()

# --- Access Control Check ---
if not check_access(allowed_roles=['doctor', 'institute']):
    st.stop()

# --- Model Loading ---
@st.cache_resource
def load_3d_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=3, out_channels=4)
    state_dict_path = r"models/segmentation_3d.pth"
    state_dict = torch.load(state_dict_path, map_location=device) # removed weights_only=True for compatibility
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_2d_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(n_channels=3, n_classes=1)
    state_dict_path = r"models/segmentation_2d.pth"
    checkpoint = torch.load(state_dict_path, map_location=device)
    
    # Check if the checkpoint is a state_dict or a full checkpoint
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model, device

# Load models
model_3d, device_3d = load_3d_model()
model_2d, device_2d = load_2d_model()

# Fix for OpenMP on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Main Page Tabs ---
tab1, tab2 = st.tabs(["3D Segmentation (NIfTI)", "2D Segmentation (JPG/PNG)"])

# --- TAB 1: 3D SEGMENTATION ---
with tab1:
    st.header("3D Volumetric Segmentation")
    with st.expander("ℹ️ How to Use This Page"):
        st.markdown("""
        This page uses a **3D U-Net** model to segment brain tumors from NIfTI files.

        **1. Required Files:** Upload three MRI sequences:
            - **T1CE (T1-weighted Contrast-Enhanced)**
            - **T2 (T2-weighted)**
            - **FLAIR (Fluid Attenuated Inversion Recovery)**
        
        **2. Optional File:** `T1 (T1-weighted)`
        **3. Output:** A downloadable 3D segmentation mask in `.nii` format.
        """)

    # Helper functions for 3D
    def load_nii(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        img = nib.load(tmp_path)
        return img.get_fdata(), img.affine

    def preprocess_images_3d(t1ce, t2, flair):
        channels = [t1ce, t2, flair]
        channels = [(ch - ch.min()) / (ch.max() - ch.min() + 1e-5) for ch in channels]
        img_array = np.stack(channels, axis=0)  # shape (C, H, W, D)
        return img_array.astype(np.float32)

    def pad_to_multiple(img_array, multiple=16):
        c, h, w, d = img_array.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        pad_d = (multiple - d % multiple) % multiple
        pad = ((0,0), (0,pad_h), (0,pad_w), (0,pad_d))
        img_array = np.pad(img_array, pad, mode='constant', constant_values=0)
        return img_array, pad_h, pad_w, pad_d

    def unpad_mask(mask, pad_h, pad_w, pad_d):
        if pad_h > 0: mask = mask[:-pad_h, :, :]
        if pad_w > 0: mask = mask[:, :-pad_w, :]
        if pad_d > 0: mask = mask[:, :, :-pad_d]
        return mask

    def segment_volume(img_array):
        img_array, pad_h, pad_w, pad_d = pad_to_multiple(img_array, multiple=16)
        tensor = torch.from_numpy(img_array).unsqueeze(0).to(device_3d)
        with torch.no_grad():
            pred = model_3d(tensor)
            mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        mask = unpad_mask(mask, pad_h, pad_w, pad_d)
        return mask

    def display_random_slice_modalities(t1ce, t2, flair, t1, mask):
        tumor_slices = np.any(mask > 0, axis=(0,1))
        slice_idx = random.choice(np.where(tumor_slices)[0]) if np.any(tumor_slices) else t1ce.shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 6), facecolor='black')
        axes = axes.ravel()
        
        modalities = {'Flair': flair, 'T1': t1, 'T1CE': t1ce, 'T2': t2, 'Predicted Mask': mask}
        cmaps = {'Flair': 'gray', 'T1': 'gray', 'T1CE': 'gray', 'T2': 'gray', 'Predicted Mask': 'Reds'}
        
        for i, (title, img) in enumerate(modalities.items()):
            ax = axes[i]
            ax.set_facecolor('black')
            ax.set_title(title, color='white')
            ax.axis('off')
            if img is not None:
                cmap = cmaps[title]
                alpha = 0.6 if title == 'Predicted Mask' else 1.0
                ax.imshow(np.rot90(img[:,:,slice_idx]), cmap=cmap, alpha=alpha)
            else:
                ax.imshow(np.zeros_like(np.rot90(flair[:,:,slice_idx])), cmap='gray')
                ax.set_title(f"{title} (Not Provided)", color='white')
        
        # Overlay Mask on T1CE
        axes[5].imshow(np.rot90(t1ce[:,:,slice_idx]), cmap='gray')
        axes[5].imshow(np.rot90(mask[:,:,slice_idx]), cmap='Reds', alpha=0.5)
        axes[5].set_title('Mask Overlay on T1CE', color='white')

        plt.tight_layout()
        st.pyplot(fig)

    def save_nii(mask, affine):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), tmp_file.name)
            return tmp_file.name

    # UI for 3D
    st.write("Upload 3 mandatory NIfTI files (t1ce, t2, flair) and 1 optional (t1).")
    col1_3d, col2_3d, col3_3d, col4_3d = st.columns(4)
    with col1_3d: t1ce_file = st.file_uploader("Upload t1ce.nii", type=["nii"])
    with col2_3d: t2_file = st.file_uploader("Upload t2.nii", type=["nii"])
    with col3_3d: flair_file = st.file_uploader("Upload flair.nii", type=["nii"])
    with col4_3d: t1_file = st.file_uploader("Upload t1.nii (optional)", type=["nii"])

    if t1ce_file and t2_file and flair_file:
        with st.spinner("Processing 3D volume... This may take a moment."):
            t1ce, affine = load_nii(t1ce_file)
            t2, _ = load_nii(t2_file)
            flair, _ = load_nii(flair_file)
            t1 = None
            if t1_file:
                t1, _ = load_nii(t1_file)

            img_array = preprocess_images_3d(t1ce, t2, flair)
            mask = segment_volume(img_array)

            st.success("Segmentation complete!")
            st.session_state.segmentation_done = True # For Stryden.ai context
            display_random_slice_modalities(t1ce, t2, flair, t1, mask)

            download_path = save_nii(mask, affine)
            with open(download_path, "rb") as f:
                st.download_button("Download Segmented Mask (.nii)", f, file_name="segmented_mask_3d.nii")


# --- TAB 2: 2D SEGMENTATION ---
with tab2:
    st.header("2D Slice Segmentation")
    with st.expander("ℹ️ How to Use This Page"):
        st.markdown("""
        This page uses a **2D U-Net** model to segment a brain tumor from a single image.

        **1. Required File:** Upload one MRI scan:
            - **JPG, JPEG, or PNG**
        
        **2. Output:** A downloadable segmentation mask in `.png` format.
        """)
    
    # Preprocessing for 2D
    transform_2d = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    def predict_2d(image, model):
        """Run inference on a 2D image."""
        original_image = np.array(image)
        
        # Preprocess
        augmented = transform_2d(image=original_image)
        input_tensor = augmented['image'].unsqueeze(0).to(device_2d)
        
        # Predict
        with torch.no_grad():
            preds = model(input_tensor)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float() # Binarize
            
        # Post-process
        mask = preds.squeeze().cpu().numpy()
        
        # Resize mask to original image size
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)
        mask_resized = np.array(mask_pil) / 255.0
        
        return original_image, mask_resized

    def create_overlay(image, mask):
        """Create an overlay of the mask on the image."""
        fig, ax = plt.subplots(facecolor='black')
        ax.imshow(image, cmap='gray')
        # Create a colored mask (e.g., red)
        colored_mask = np.zeros((*mask.shape, 4)) # RGBA
        colored_mask[mask > 0.5] = [1, 0, 0, 0.6] # Red with 60% alpha
        
        ax.imshow(colored_mask)
        ax.axis('off')
        
        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
        buf.seek(0)
        plt.close(fig)
        return buf
    
    def save_mask_to_buffer(mask):
        """Saves the binary mask to a buffer for downloading."""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    # UI for 2D
    uploaded_file_2d = st.file_uploader("Upload an MRI scan (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file_2d:
        image = Image.open(uploaded_file_2d).convert("RGB")
        
        with st.spinner("Segmenting 2D image..."):
            original_image, mask = predict_2d(image, model_2d)
            overlay_buffer = create_overlay(original_image, mask)
            
            st.success("2D Segmentation complete!")
            
            # Store for dashboard
            if 'seg_2d_images' not in st.session_state:
                st.session_state.seg_2d_images = []
            st.session_state.seg_2d_images.append({
                "input": uploaded_file_2d,
                "output_mask": save_mask_to_buffer(mask),
                "output_overlay": overlay_buffer
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Uploaded Image", use_column_width=True)
            with col2:
                st.image(overlay_buffer, caption="Predicted Segmentation Mask (Overlay)", use_column_width=True)

            # Download button
            mask_buf_for_download = save_mask_to_buffer(mask)
            st.download_button(
                label="Download Mask (.png)",
                data=mask_buf_for_download,
                file_name=f"segmented_mask_2d_{uploaded_file_2d.name}.png",
                mime="image/png"
            )
