import streamlit as st
from memory import ask_query

def ask(query):
    return ask_query(query)

st.title("The Mismatched APP")

query = st.chat_input("Waiting for your message")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

avatar_map = {
    "user": "https://raw.githubusercontent.com/ishebee/The-Mismatched-APP/refs/heads/main/avatars/girl.png",
    "Zwan": "https://raw.githubusercontent.com/ishebee/The-Mismatched-APP/refs/heads/main/avatars/boy.png",
    "assistant": "https://raw.githubusercontent.com/ishebee/The-Mismatched-APP/refs/heads/main/avatars/boy.png",
}

for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role, avatar=avatar_map.get(role, None)):
        if message.get("type") == "image":
            st.image(message["content"], width=300)
        else:
            st.markdown(message["content"])

if query:
    with st.chat_message("user",avatar="https://raw.githubusercontent.com/ishebee/The-Mismatched-APP/refs/heads/main/avatars/girl.png"):
        st.markdown(query)
    st.session_state.messages.append({"role":"user", "content":query})

    response, image = ask(query)
    with st.chat_message("Zwan",avatar="https://raw.githubusercontent.com/ishebee/The-Mismatched-APP/refs/heads/main/avatars/boy.png"):
        st.markdown(response)
    st.image(image, width = 300)
    st.session_state.messages.append({"role": "assistant", "type": "image", "content": image})
    st.session_state.messages.append({"role": "assistant", "content": response})