import streamlit as st
import os
from timeit import default_timer as timer
import sys
sys.path.append("./_pages")
import ctif_readhybrid 
from streamlit_float import *
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

st.set_page_config(page_title="GraphRAG Bot",layout="wide")
set_llm_cache(InMemoryCache())

os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"] 
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"] 
os.environ["OPENAI_MODEL_NAME"] = st.secrets["OPENAI_MODEL_NAME"] 
os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"] 
os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"] 

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "hybrid_messages" not in st.session_state:
    st.session_state.hybrid_messages = []

if "topic_messages" not in st.session_state:
    st.session_state.topic_messages = []

if "topic_hybrid_messages" not in st.session_state:
    st.session_state.topic_hybrid_messages = []

st.subheader("Let's start exploring your data")

# Display chat messages from history on app rerun   
container_tab1 = st.container(height=400)    
for message in st.session_state.hybrid_messages:
    print("Message: ", message)
    with container_tab1.chat_message(message["role"]):
        # new addition
        if message["role"] == "assistant":
            st.write(message["content"])
        if message["role"] == "user":
            st.write(message["content"])   

if prompt := st.chat_input("Start chatting!"):

    st.session_state.messages.append({"role": "user", "content": prompt, "mode": "prompt"})
    with container_tab1.chat_message("user"):
        st.write(prompt)

    with container_tab1.chat_message("assistant"):
        res_hybrid = ctif_readhybrid.query(prompt)
        print("RES_HYBRID: ", res_hybrid)
        st.write(str(res_hybrid))
        st.session_state.messages.append({"role": "assistant", "content": res_hybrid})
st.markdown("""
<style>
table {
    width: 100%;
    border-collapse: collapse;
    border: none !important;
    font-family: "Source Sans Pro", sans-serif;
    color: rgba(49, 51, 63, 0.6);
    font-size: 0.9rem;
}

tr {
    border: none !important;
}

th {
    text-align: center;
    colspan: 3;
    border: none !important;
    color: #0F9D58;
}

th, td {
    padding: 2px;
    border: none !important;
}
</style>

<table>
<tr>
    <th colspan="3">Sample Questions</th>
</tr>
<tr>
    <td>What is defendant's speech at the Tablik Akbar act?</td>
    <td>Who is Munarman as a  defendant?</td>
    <td>Who is MOHAMMAD AKBAR MUSLIM as a witness?</td>
</tr>
<tr>
    <td>What has been talked about oil companies??</td>
    <td>Where did Assult happen?</td>
            
</tr>
<tr>
    <td></td>
    <td></td>
</tr>
</table>
""", unsafe_allow_html=True)

