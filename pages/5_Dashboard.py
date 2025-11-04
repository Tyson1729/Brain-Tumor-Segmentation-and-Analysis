import streamlit as st
import pandas as pd

st.set_page_config(page_title="Session Dashboard", layout="wide")
st.title("📊 Session Dashboard")

# --- Helpers & Access Control ---
def logout():
    """Clears all session data and logs the user out."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    st.experimental_rerun()

def check_access(allowed_roles):
    """Checks if the user is logged in and has the correct role."""
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in first from the 'User's Credentials' page.")
        return False
    
    role = st.session_state.get('role', 'patient')
    if role not in allowed_roles:
        st.error("Access Denied: Your user role does not have permission to view this page.")
        st.error("Only 'Doctor' or 'Institute' roles can access the Dashboard.")
        return False
    return True

# Add logout button
st.markdown('<div style="position: absolute; top: 10px; right: 10px;"><a href="#" id="logout-button"></a></div>', unsafe_allow_html=True)
if st.button("Log Out"):
    logout()

# --- Access Control Check ---
if not check_access(allowed_roles=['doctor', 'institute']):
    st.stop()

st.markdown("---")
st.info("This page displays all data collected during the current session.")

# --- Display User Data ---
st.header("Current User Information")
if 'user_data' in st.session_state:
    st.json(st.session_state.user_data)
else:
    st.warning("No user data found in session.")

# --- Display Session Log ---
st.header("Session Activity Log")
if 'session_log' in st.session_state and st.session_state.session_log:
    log_df = pd.DataFrame(st.session_state.session_log)
    st.dataframe(log_df)
else:
    st.info("No activities (like classification or login) have been logged yet.")

# --- Display 2D Segmentation Results ---
st.header("2D Segmentation Results")
if 'seg_2d_images' in st.session_state and st.session_state.seg_2d_images:
    st.info(f"Found {len(st.session_state.seg_2d_images)} 2D segmentation result(s).")
    
    for i, result in enumerate(st.session_state.seg_2d_images):
        st.markdown(f"---")
        st.subheader(f"Result {i+1} (File: {result['input'].name})")
        
        col1, col2 = st.columns(2)
        
        # Reset buffers
        result['input'].seek(0)
        result['output_overlay'].seek(0)
        result['output_mask'].seek(0)
        
        with col1:
            st.image(result['input'], caption="Original Input", use_column_width=True)
        with col2:
            st.image(result['output_overlay'], caption="Segmented Overlay", use_column_width=True)

        st.download_button(
            label=f"Download Input {i+1} (.jpg)",
            data=result['input'],
            file_name=f"input_{i+1}_{result['input'].name}",
            mime="image/jpeg"
        )
        st.download_button(
            label=f"Download Mask {i+1} (.png)",
            data=result['output_mask'],
            file_name=f"mask_{i+1}_{result['input'].name}.png",
            mime="image/png"
        )
        
else:
    st.info("No 2D segmentations have been performed in this session.")
