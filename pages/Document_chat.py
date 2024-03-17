# Import necessary modules
import pandas as pd
import streamlit as st 

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# My OpenAI Key
import os
os.environ['OPENAI_API_KEY'] = "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"

# Page configuration for Simple PDF App
st.set_page_config(
    page_title="Document Q&A with AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
st.sidebar.subheader("Setup")
OPENAI_API_KEY = "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"
st.sidebar.subheader("Model Selection")
llm_model_options = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4']  # Add more models if available
model_select = st.sidebar.selectbox('Select LLM Model:', llm_model_options, index=0)





if "conversation" not in st.session_state:
    st.session_state.conversation = None

st.markdown(f"""## Document Chat 📑 <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)



# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.1, model_name=model_select),
        retriever=vectorstore,
        get_chat_history=lambda h : h,
        memory=memory
    )
    return conversation_chain

clear_history = st.sidebar.button("Clear conversation history")
on = st.toggle('Start Chat')
if on:
    embeddings = OpenAIEmbeddings()

    db = Chroma (persist_directory="/home/abhishek/Desktop/personal_projects/Chat_App/pages/db", embedding_function = embeddings)

    vectorstore = db.as_retriever()
    # Create FAISS Vector Store of PDF Docs

    st.session_state.conversation = get_conversation_chain(vectorstore)
            


# Initialize chat history in session state for Document Analysis (doc) if not present
if 'doc_messages' not in st.session_state or clear_history:
    # Start with first message from assistant
    st.session_state['doc_messages'] = [{"role": "assistant", "content": "Query your documents"}]
    st.session_state['chat_history'] = []  # Initialize chat_history as an empty list

# Display previous chat messages
for message in st.session_state['doc_messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# If user provides input, process it
if user_query := st.chat_input("Enter your query here"):
    # Add user's message to chat history
    st.session_state['doc_messages'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Generating response..."):
        # Check if the conversation chain is initialized
        if 'conversation' in st.session_state:
            st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            # Process the user's message using the conversation chain
            result = st.session_state.conversation({
                "question": user_query, 
                "chat_history": st.session_state['chat_history']})
            response = result["answer"]
            # Append the user's question and AI's answer to chat_history
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": response
            })
        else:
            response = "Please upload a document first to initialize the conversation chain."
        
        # Display AI's response in chat format
        with st.chat_message("assistant"):
            st.write(response)
        # Add AI's response to doc_messages for displaying in UI
        st.session_state['doc_messages'].append({"role": "assistant", "content": response})
