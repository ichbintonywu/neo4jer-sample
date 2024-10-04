import os
from neo4j import GraphDatabase
import sys
sys.path.append("./_drivers")
from neo4j_handler import exec_cypher_query
import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import  List
import os
from openai import AzureOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_openai import  AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from neo4j import GraphDatabase
import streamlit as st
import ast
from langchain.schema import HumanMessage
from  langchain_ollama  import OllamaLLM
from langchain_ollama import ChatOllama


st.set_page_config(page_title="Activity Analysis with Entity Resolution",layout="wide")

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

api_version="2024-06-01"
endpoint="https://sabikiopenai.openai.azure.com/"

llm = AzureChatOpenAI(
    api_version=api_version,
    # model="sabikiGPT4Deployment",
    model="gpt4oDeploy",
    api_key="a9841f320eb64c87a80c0f2d39be383b",
    azure_endpoint=endpoint,
    temperature=0.3,
    max_tokens = 3000
)
llamallm = OllamaLLM(
    model="llama3.1",
    temperature=0,
    base_url='52.163.241.32'
)
# llamallm = ChatOllama(
#     model="llama3.1",
#     temperature=0,
#     base_url='52.163.241.32'
# )

def chat_request(start_phrase):
    messages = [HumanMessage(content=start_phrase)]
    response = llm(messages)
    print(start_phrase+response.content)
    return response.content

def chat_llama_request(start_phrase):
    print(start_phrase)
    messages = [
        (
            "system",
            "You are a helpful assistant that summarize a person's activities in time order.",
        ),
        ("human", start_phrase),
    ]
    ai_msg = llamallm.invoke(messages)
    return ai_msg

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
match path = (n2)-[:ENTITY_RESOLUTION]-(n:Person)--(a:Activity), (n)--(e:Time),(n)--(d:Date)
with n2.id as similarPerson, n.id as person,COLLECT(DISTINCT d.id) as date, collect(distinct e.id) as timelines, collect(distinct a.id) as activities limit 5
with person,similarPerson,date,timelines,activities,apoc.coll.zip(date,activities) as date_activities
return distinct person
"""

get_Person_List =  exec_cypher_query(getPerson_str)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_colwidth', 5000) 

ndarray_results = np.array(get_Person_List, dtype=object)
dfperson_list = pd.DataFrame(ndarray_results[:,0], columns=["person"])

with st.form("Person_Email_form",clear_on_submit=False):
    col1,col2 =st.columns(2)
    col2.selectbox("Select a Person:", dfperson_list,key="dfxbar_selection")
    
    "---"
    submitted_selected_Person = st.form_submit_button("Display high Entity Resulution score(FuzzyMatch=true OR levenshteinSimilarity>0.8 OR sorensenDiceSimilarity>0.8 OR phoneticMatch=true) people and their activities")
    if submitted_selected_Person:
        cur_person = str(st.session_state["dfxbar_selection"])
        getPersonEmail_str=f"""
        match path = (n2)-[:ENTITY_RESOLUTION]-(n:Person)--(a:Activity), (n)--(e:Time),(n)--(d:Date) where n.id='{cur_person}'
        with n2.id as similarPerson, n.id as person,COLLECT(DISTINCT d.id) as date, collect(distinct e.id) as timelines, collect(distinct a.id) as activities limit 5
        with person,similarPerson,date,timelines,activities,apoc.coll.zip(date,activities) as date_activities
        return person, similarPerson,date_activities"""
        print(getPersonEmail_str)
        get_Person_Email =  exec_cypher_query(getPersonEmail_str)
        ndarray_results2 = np.array(get_Person_Email, dtype=object)
        dfpersonemail = pd.DataFrame(ndarray_results2, columns=["person","other people under Entity Resolution","date_activities"])
        print(dfpersonemail["date_activities"])
        st.dataframe(dfpersonemail,use_container_width=True, hide_index=True
                     )
        
        myPrompt = f"""Summarize a person -{dfpersonemail["person"]}'s actitivies based on the information below,before summarizing, please rephrase violence words, and nouns like hate, make the answer as comprehensive as possible \n {dfpersonemail["date_activities"]} """

        response = chat_request(myPrompt)
        st.write(response)
        # "---"
        # print("--------")
        # merged_column = ' '.join(map(str,dfpersonemail['date_activities']))
        # myPrompt2 = f"""Summarize a person -{dfpersonemail["person"]}'s actitivies based on the information below,before summarizing, please rephrase violence words, and nouns like hate, make the answer as comprehensive as possible \n {merged_column} """

        # print(merged_column)

        # llama_response = chat_llama_request(myPrompt2)
        # st.write (llama_response)
        