import os
from neo4j import GraphDatabase
import sys
sys.path.append("./_drivers")
from neo4j_handler import exec_cypher_query
import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai.chat_models.base import BaseChatOpenAI
from typing import  List
import os
from openai import AzureOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_openai import  AzureChatOpenAI
from neo4j import GraphDatabase
import streamlit as st
from langchain.schema import HumanMessage

st.set_page_config(page_title="Shared Attribute Analysis",layout="wide")

NEO4J_HOST = st.secrets["NEO4J_AURA_CTIF"]+":"+st.secrets["NEO4J_PORT"] 
NEO4J_USER = st.secrets["NEO4J_AURA_CTIF_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_AURA_CTIF_PASSWORD"]
NEO4J_DATABASE = 'neo4j'

AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]
os.environ["NEO4J_URI"] = NEO4J_HOST
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"] 
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"] 
os.environ["OPENAI_MODEL_NAME"] = st.secrets["OPENAI_MODEL_NAME"] 
os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"] 
os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"] 

graph = Neo4jGraph(database=NEO4J_DATABASE)
AUTH = (NEO4J_USER,NEO4J_PASSWORD)
driver = GraphDatabase.driver(
NEO4J_HOST, auth=AUTH)

AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]

client = AzureOpenAI(
    api_key = os.environ["AZURE_API_KEY"],  
    api_version = os.environ["AZUER_VERSION"],
    azure_endpoint = os.environ["AZURE_ENDPOINT"],
    ) 

deployment_name=AZURE_DEPLOYMENT_NAME
api_version= AZURE_VERSION
endpoint= AZURE_ENDPOINT

llm = AzureChatOpenAI(
    api_version = api_version,
    model= AZURE_DEPLOYMENT,
    api_key = AZURE_KEY,
    azure_endpoint=endpoint,
    temperature=0.1,
    max_tokens = 3000
)

def chat_request(start_phrase):
    messages = [HumanMessage(content=start_phrase)]

# Get a response from the model
    response = llm(messages)
    # response = client.completions.create(model="gpt-4-32k",prompt=start_phrase, max_tokens=2000)
    print(start_phrase+response.content)
    return response.content

def exec_cypher_query(qry_str):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.execute_read(do_cypher_tx,qry_str)
        return result

def do_cypher_tx(tx,cypher):
    results = tx.run(cypher)
    values = []
    for record in results:
        values.append(record.values())
    return values

getPerson_str="""
match path = (n:Person)--(e:Email)--(n2:Person)--(a:Activity) where id(n)<>id(n2)  and e.id contains("@")
with n.id as person, collect(distinct n2.id) as otherPersons, collect(distinct a.id) as activities, e.id as email
return person"""

get_Person_List =  exec_cypher_query(getPerson_str)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

ndarray_results = np.array(get_Person_List, dtype=object)
dfperson_list = pd.DataFrame(ndarray_results[:,0], columns=["person"])

with st.form("Person_Email_form",clear_on_submit=False):
    col1,col2 =st.columns(2)
    col2.selectbox("Select a Person:", dfperson_list,key="dfxbar_selection")
    
    "---"
    submitted_selected_Person = st.form_submit_button("Display Shared email address and their activities")
    if submitted_selected_Person:
        cur_person = str(st.session_state["dfxbar_selection"])
        getPersonEmail_str=f"""
        match path = (n:Person)--(e:Email)--(n2:Person)--(a:Activity) where id(n)<>id(n2)  and e.id contains("@") and n.id = '{cur_person}'
        with n.id as person, collect(distinct n2.id) as otherPersons, collect(distinct a.id) as activities, e.id as email
        return person, otherPersons, email,activities"""
        print(getPersonEmail_str)
        get_Person_Email =  exec_cypher_query(getPersonEmail_str)
        ndarray_results2 = np.array(get_Person_Email, dtype=object)
        dfpersonemail = pd.DataFrame(ndarray_results2, columns=["person","other people","shared email address","activities"])
        print(dfpersonemail["activities"])
        st.dataframe(dfpersonemail,use_container_width=True, hide_index=True
                     )
        
        myPrompt = f"""Summarize a person -{dfpersonemail["person"]}'s actitivies based on the information below and Make the answer as comprehensive as possible \n {dfpersonemail["activities"]} """

        response = chat_request(myPrompt)
        st.write(response)
        
