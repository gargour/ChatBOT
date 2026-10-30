import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

def get_chatbot_chain():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY non trouvée dans les variables d'environnement.")
    
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=api_key,
        temperature=0.7
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant virtuel intelligent, utile et sympathique."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return conversational_chain