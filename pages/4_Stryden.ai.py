import streamlit as st
import google.generativeai as genai

# --- Page Configuration & Helpers ---
st.set_page_config(page_title="Stryden.ai Assistant", page_icon="💡", layout="wide")
st.title("💡 Stryden AI - Your Medical Imaging Assistant")

def logout():
    """Clears all session data and logs the user out."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    # st.experimental_rerun()

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

# Initialize Session State
def init_chat_session_state():
    defaults = {
        "messages": [],
        "api_key": st.session_state.get("api_key"), # Persist API key if already set
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_chat_session_state()

# Sidebar for API Key
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input(
        "Enter your Google Gemini API Key:",
        type="password",
        help="Get your key from Google AI Studio. It is not stored after your session ends."
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("API Key set! You can now chat.")

# Helper Function to Create the System Prompt
def create_system_prompt():
    """Creates a contextual prompt for the Gemini model based on session state."""
    base_prompt = """
    You are Stryden.ai, a specialized AI assistant integrated into a Brain Tumor Analysis Streamlit application.
    Your role is to provide clear, helpful, and safe information related to brain tumors, medical imaging, and the application's functionality.
    Your tone should be professional, empathetic, and educational.
    
    IMPORTANT: You MUST NOT provide medical advice, diagnosis, or treatment plans.
    You MUST ALWAYS remind the user to consult a qualified healthcare professional for any medical concerns.
    """

    # Check for context from other pages
    context = ""
    tumor_type = st.session_state.get('tumor_class')
    seg_done = st.session_state.get('segmentation_done')

    if tumor_type and tumor_type != 'no_tumor':
        context += f"""
        CONTEXT: The user has just used the 'Brain Tumor Classification' tool, which predicted a '{tumor_type.replace('_', ' ').title()}' tumor.
        The user (Role: {st.session_state.get('role', 'unknown')}) may ask what this means. 
        Explain what a {tumor_type.replace('_', ' ').title()} tumor is in general terms (e.g., where it originates, common characteristics).
        Preface any explanation with: "Based on the classification result, here is some general information about {tumor_type.replace('_', ' ').title()} tumors. Please remember, this is not a diagnosis..."
        """
    elif tumor_type == 'no_tumor':
        context += """
        CONTEXT: The user has just used the 'Brain Tumor Classification' tool, and it predicted 'No Tumor'.
        Be encouraging and answer any general questions they have about brain health or the other features of the application.
        """

    if seg_done:
        context += """
        ADDITIONAL CONTEXT: The user (Role: {st.session_state.get('role', 'unknown')}) has also recently performed a 3D tumor segmentation.
        They might ask about what segmentation is, why it's useful (e.g., for surgical planning, monitoring tumor volume), or about the different MRI sequences (T1, T2, FLAIR).
        """

    if not context:
        context = """
        CONTEXT: The user (Role: {st.session_state.get('role', 'unknown')}) has not yet used the classification or segmentation tools.
        Answer their general questions about the application, different types of brain tumors, or the underlying AI technology (like CNNs and U-Nets).
        """

    return base_prompt + context

# --- Main Chat Interface ---
if not st.session_state.api_key:
    st.info("👋 Welcome! Please add your Gemini API key in the sidebar to activate the assistant.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about brain tumors, segmentation, or your results..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Stryden is thinking..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model_name = 'gemini-1.5-flash'
                try:
                    genai.GenerativeModel('gemini-2.0-flash')
                    model_name = 'gemini-2.0-flash'
                except Exception:
                    pass # Fallback to 1.5 flash

                model = genai.GenerativeModel(model_name)
                
                system_prompt = create_system_prompt()
                
                # Create history for the model
                model_history = []
                for msg in st.session_state.messages[:-1]: # All except the last user prompt
                    model_history.append({"role": msg["role"], "parts": [msg["content"]]})

                # Start a chat session with history
                chat_session = model.start_chat(history=model_history)
                
                # Send the new prompt
                full_prompt = system_prompt + "\n\nUser's question: " + prompt
                response = chat_session.send_message(full_prompt)
                response_text = response.text

            except Exception as e:
                response_text = f"An error occurred: {e}. Please check your API key and try again."

        st.markdown(response_text)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
