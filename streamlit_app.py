import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("AI Chat with Multiple PDFs")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    files = [("files", file) for file in uploaded_files]
    response = requests.post(f"{BACKEND_URL}/upload-pdfs/", files=files)
    if response.status_code == 200:
        st.success("PDFs processed successfully!")
    else:
        st.error("Error processing PDFs")

user_query = st.text_input("Ask a question about the uploaded PDFs:")
if user_query:
    response = requests.post(f"{BACKEND_URL}/query/", json={"question": user_query})
    if response.status_code == 200:
        st.write("AI Response:", response.json()["response"])
    else:
        st.error("Error getting response")