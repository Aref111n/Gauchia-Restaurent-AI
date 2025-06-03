import streamlit as st
import requests

st.set_page_config(page_title="FunnyAI Chat", layout="centered")

# --- Function to generate response ---
def generate_text(qstn):
    response = requests.post("http://127.0.0.1:8000/api/chat/",
                             json={
                                 "message": qstn,
                                 "role": "user",
                                 "conversation_id": "string"
                             })
    if response.status_code == 200:
        return response.json().get('response', 'No response received.')
    else:
        return "Error: Unable to generate response."

# --- Initialize session state ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Page title ---
st.title("Gauchia Restaurent AI")
st.subheader("Your premium food choice in Digital Bangladesh 2050")

# --- Display previous chat messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input box ---
if user_input := st.chat_input("Ask your queries about our recipe/price/availability or anything!!!"):
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display AI response
    with st.chat_message("assistant"):
        response = generate_text(user_input)
        st.markdown(response)

    # Add assistant response to session
    st.session_state.messages.append({"role": "assistant", "content": response})
