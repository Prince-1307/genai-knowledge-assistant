import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"  # local for now, will change to deployed URL later

st.set_page_config(page_title="Personal Knowledge Assistant", page_icon="🧠", layout="wide")
st.title("Personal Knowledge Assistant")
st.caption("Upload your PDFs and chat with them using AI")

# --- Sidebar ---
with st.sidebar:
    st.header("Documents")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Uploading and indexing..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            )
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ {data['filename']} indexed ({data['chunks_added']} chunks)")
            else:
                st.error(f"Upload failed: {response.text}")

    # Document list
    st.subheader("Indexed Documents")
    try:
        docs_response = requests.get(f"{API_URL}/documents")
        if docs_response.status_code == 200:
            docs = docs_response.json()["documents"]
            if docs:
                for doc in docs:
                    st.write(f"📄 {doc}")
            else:
                st.write("No documents yet — upload one above")
    except:
        st.write("API not reachable")

    # Reset memory button
    st.divider()
    if st.button("🔄 Reset Memory"):
        response = requests.post(f"{API_URL}/reset")
        if response.status_code == 200:
            st.success("Memory cleared!")
            st.session_state.messages = []

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call API and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Could not reach API: {e}")