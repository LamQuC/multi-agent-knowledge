import streamlit as st
import requests
import os

st.title("Multi-Agent Knowledge Demo")

agent = st.selectbox("Select agent", ["knowledge", "explain", "code"])
query = st.text_area("Enter your query")

if st.button("Send"):
    api_url = os.getenv("API_URL", "http://backend:8000/chat")
    payload = {"query": query}
    try:
        res = requests.post(api_url, json=payload, timeout=60)
        res.raise_for_status()
        st.write(res.json())
    except Exception as e:
        st.error(f"Error: {e}")
