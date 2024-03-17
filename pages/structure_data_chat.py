
# from langchain.llms import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
# from pandasai.callbacks import StdoutCallback
from pandasai.llm import OpenAI
import streamlit as st
import pandas as pd
import os

from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
# from pandasai_app.components.faq import faq
import logging
import sys
from IPython.display import Markdown, display

import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from llama_index.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
import logging
import sys
from IPython.display import Markdown, display

import pandas as pd
from llama_index.query_engine import PandasQueryEngine



# My OpenAI Key
import os
os.environ['OPENAI_API_KEY'] = "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"

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="Chat-Data")
st.title("Chat with Data")

uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

if uploaded_file:
    df = load_data(uploaded_file)

openai_api_key = "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"

chat_data = st.sidebar.selectbox("Choose a Backend", ['pandasai', 'langchain'])



if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if chat_data == "pandasai":
        #PandasAI OpenAI Model
        llm = OpenAI(api_token=openai_api_key)
        # llm = OpenAI(api_token=openai_api_key)

        sdf = SmartDataframe(df, config = {"llm": llm,
                                            "enable_cache": False,
                                            "conversational": True,
                                            })

        with st.chat_message("assistant"):
            response = sdf.chat(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    
    if chat_data == "langchain":

        llm = ChatOpenAI(
            temperature=0, model="gpt-4-0613", openai_api_key=openai_api_key, streaming=True
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)