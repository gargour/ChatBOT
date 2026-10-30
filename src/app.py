import streamlit as st
from chatbot import get_chatbot_chain

st.set_page_config(page_title="Chatbot LangChain", page_icon="🤖", layout="centered")

st.title("🤖 Assistant Chatbot Intelligent")
st.write("Discutez en temps réel avec votre assistant propulsé par LangChain et NVIDIA NIM.")

if "chain" not in st.session_state:
    st.session_state.chain = get_chatbot_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Écrivez votre message ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            try:
                config = {"configurable": {"session_id": "default_user_session"}}
                response = st.session_state.chain.invoke({"input": prompt}, config=config)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Une erreur s'est produite : {e}")