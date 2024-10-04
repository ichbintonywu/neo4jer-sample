import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import sys
sys.path.append("./_drivers")
from neo4j_handler import exec_cypher_query

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_colwidth', 5000) 

NEO4J_HOST = st.secrets["NEO4J_AURA_CTIF"]+":"+st.secrets["NEO4J_PORT"] 
NEO4J_USER = st.secrets["NEO4J_AURA_CTIF_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_AURA_CTIF_PASSWORD"]
NEO4J_DATABASE = 'neo4j'

AUTH = (NEO4J_USER,NEO4J_PASSWORD)
driver = GraphDatabase.driver(
NEO4J_HOST, auth=AUTH)

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

getPersonList_str="""
match path= (n:Person)--(x)--(n2:Person) where n.id contains ('Munarman') and EXISTS{(n)-[:ENTITY_RESOLUTION]-(n2)} and not x:Chunk return distinct n.id as person
"""

get_Person_List =  exec_cypher_query(getPersonList_str)

ndarray_results = np.array(get_Person_List, dtype=object)
dfperson_list = pd.DataFrame(ndarray_results, columns=["person"])
myBloomSearchbox = dfperson_list["person"]

st.set_page_config(page_title="Bloom Visualiztion on Aura",layout="wide")

with st.form("BloomList",clear_on_submit=False):
    col1,col2,col3 =st.columns(3)
    col2.selectbox("Select a person:", myBloomSearchbox,key="person1_selection")
    col3.selectbox("Select another person with first person's ER:", myBloomSearchbox,key="person2_selection")
    "---"
    submitted_bloom_selection = st.form_submit_button("Display Bloom Result")
    if submitted_bloom_selection:
        st.info("The shared profile for two similar people is shown below")
        actual_phrase1 = str(st.session_state["person1_selection"]).replace(" ","%20")
        actual_phrase2 = str(st.session_state["person2_selection"]).replace(" ","%20")
        iframe_src =f"""https://bloom.neo4j.io/index.html?connectURL=0a10ec94.databases.neo4j.io&run=true&search=share profile for person:{actual_phrase1} and {actual_phrase2}"""
        iframe_src2 ="https://bloom.neo4j.io/index.html?connectURL=c2f7f38c.databases.neo4j.io&run=true&search=clear"
        components.iframe(iframe_src, height=800, scrolling=True)


