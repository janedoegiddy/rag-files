# frontend/app.py
import streamlit as st
import requests
import os

# --- CONFIGURATION ---
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# --- UI SETUP ---
st.set_page_config(page_title="Document Q&A", page_icon="🤖")
st.title("📄 Document Q&A")

# --- SESSION STATE INITIALIZATION ---
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = ""
# ADD THIS: To store the user's document list
if "user_documents" not in st.session_state:
    st.session_state.user_documents = []

# --- HELPER FUNCTION TO FETCH DOCUMENTS ---
def fetch_user_documents(user_id):
    """Calls the backend to get the list of documents for a user."""
    if not user_id:
        st.session_state.user_documents = []
        return
    try:
        response = requests.get(f"{BACKEND_URL}/documents/{user_id}")
        if response.status_code == 200:
            st.session_state.user_documents = response.json().get("documents", [])
        else:
            st.error(f"Failed to fetch documents: {response.status_code}")
            st.session_state.user_documents = []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        st.session_state.user_documents = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("Setup")
    user_id = st.text_input("Enter your User ID", value=st.session_state.user_id)

    # Detect if the user ID has changed
    if user_id and user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        # Reset state for the new user
        st.session_state.chat_history = []
        st.session_state.processed_filename = ""
        fetch_user_documents(user_id)
        st.rerun() # Rerun the app to reflect the new user's state

    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file_uploader")

    if uploaded_file is not None and uploaded_file.name != st.session_state.processed_filename:
        if not st.session_state.user_id:
            st.warning("Please enter and set a User ID before uploading.")
        else:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    form_data = {"user_id": st.session_state.user_id}
                    response = requests.post(f"{BACKEND_URL}/upload", data=form_data, files=files)

                    if response.status_code == 200:
                        st.success(response.json().get("status", "File processed successfully."))
                        st.session_state.processed_filename = uploaded_file.name
                        # Refresh the document list after a successful upload
                        fetch_user_documents(st.session_state.user_id)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        st.session_state.processed_filename = ""
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the backend: {e}")
                    st.session_state.processed_filename = ""

    # --- NEW SECTION TO DISPLAY USER DOCUMENTS ---
    if st.session_state.user_id:
        st.header("Your Documents")
        if st.button("Refresh List"):
            with st.spinner("Refreshing..."):
                fetch_user_documents(st.session_state.user_id)

        if st.session_state.user_documents:
            for doc_name in st.session_state.user_documents:
                st.write(f"📄 {doc_name}")
        else:
            st.info("No documents found for this user.")


# --- MAIN CHAT INTERFACE (No changes needed here) ---
st.header("Chat with your Document")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    if not st.session_state.user_id:
        st.warning("Please enter your User ID in the sidebar to start chatting.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "user_id": st.session_state.user_id,
                        "question": prompt,
                        "chat_history": [(msg["content"], "") if msg["role"] == "user" else ("", msg["content"]) for msg in st.session_state.chat_history if msg['content']]
                    }
                    response = requests.post(f"{BACKEND_URL}/ask", json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer found.")
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                        with st.expander("See source documents"):
                            for doc in data.get("source_documents", []):
                                st.write(doc)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error."})
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the backend: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I couldn't connect to the backend."})
