import streamlit as st
import pandas as pd

st.set_page_config(page_title="User Credentials", layout="wide")
st.title("👤 User Credentials & Session")
st.markdown("---")
st.markdown("Please log in to your session. Your role will determine which pages you can access.")

# Initialize session state keys
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user_data = {}
    st.session_state.session_log = [] # To store all data for the dashboard

def logout():
    """Clears all session data and logs the user out."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    st.success("You have been logged out.")

# --- Main Page Logic ---
if st.session_state.logged_in:
    st.success(f"You are currently logged in as **{st.session_state.user_data.get('Name', 'User')}** (Role: **{st.session_state.role}**).")
    st.button("Log Out", on_click=logout)
    
    st.subheader("Current Session Data:")
    st.json(st.session_state.user_data)

else:
    tab1, tab2, tab3 = st.tabs(["Patient / Consumer", "Doctor / Professional", "Medical Institute"])

    with tab1:
        st.header("Patient Registration")
        with st.form("patient_form"):
            name = st.text_input("Name*")
            age = st.number_input("Age*", min_value=0, max_value=120, step=1)
            gender = st.selectbox("Gender*", ["Male", "Female", "Other", "Prefer not to say"])
            patient_id = st.text_input("Patient ID (Optional)")
            height = st.number_input("Height (cm) (Optional)", min_value=0.0, step=0.1, format="%.1f")
            weight = st.number_input("Weight (kg) (Optional)", min_value=0.0, step=0.1, format="%.1f")
            
            submitted = st.form_submit_button("Log In as Patient")
            if submitted:
                if not name or not age or not gender:
                    st.error("Please fill in all required fields (*).")
                else:
                    st.session_state.logged_in = True
                    st.session_state.role = "patient"
                    st.session_state.user_data = {
                        "Name": name,
                        "Age": age,
                        "Gender": gender,
                        "Patient ID": patient_id,
                        "Height (cm)": height,
                        "Weight (kg)": weight,
                        "Role": "Patient"
                    }
                    st.session_state.session_log.append({"user": name, "action": "Login", "role": "Patient"})
                    st.success(f"Welcome, {name}! You are logged in as a Patient.")
                    st.rerun()

    with tab2:
        st.header("Doctor Registration")
        with st.form("doctor_form"):
            doc_name = st.text_input("Doctor's Name*")
            doc_id = st.text_input("Doctor's ID*")
            institute_name = st.text_input("Institute Name*")
            patient_id_for_doc = st.text_input("Patient ID to access*")
            
            submitted = st.form_submit_button("Log In as Doctor")
            if submitted:
                if not doc_name or not doc_id or not institute_name or not patient_id_for_doc:
                    st.error("Please fill in all required fields (*).")
                else:
                    st.session_state.logged_in = True
                    st.session_state.role = "doctor"
                    st.session_state.user_data = {
                        "Name": doc_name,
                        "Doctor ID": doc_id,
                        "Institute Name": institute_name,
                        "Accessing Patient ID": patient_id_for_doc,
                        "Role": "Doctor"
                    }
                    st.session_state.session_log.append({"user": doc_name, "action": "Login", "role": "Doctor"})
                    st.success(f"Welcome, Dr. {doc_name}! You are logged in as a Doctor.")
                    st.rerun()

    with tab3:
        st.header("Medical Institute Access")
        with st.form("institute_form"):
            institute_name_inst = st.text_input("Institute Name*")
            region = st.text_input("Region*")
            patient_id_for_inst = st.text_input("Patient ID to access*")

            submitted = st.form_submit_button("Log In as Institute")
            if submitted:
                if not institute_name_inst or not region or not patient_id_for_inst:
                    st.error("Please fill in all required fields (*).")
                else:
                    st.session_state.logged_in = True
                    st.session_state.role = "institute"
                    st.session_state.user_data = {
                        "Institute Name": institute_name_inst,
                        "Region": region,
                        "Accessing Patient ID": patient_id_for_inst,
                        "Role": "Institute"
                    }
                    st.session_state.session_log.append({"user": institute_name_inst, "action": "Login", "role": "Institute"})
                    st.success(f"Welcome, {institute_name_inst}! You are logged in as an Institute.")
                    st.rerun()
