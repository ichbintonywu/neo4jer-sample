import streamlit as st
import streamlit.components.v1 as components

myBloomlist= ["case information", "shared email address paths", "shared phone and activities", "activity with ENTITY_RESOLUTION"]

st.set_page_config(page_title="Bloom Visualiztion on Aura",layout="wide")

with st.form("BloomList",clear_on_submit=False):
    col1,col2 =st.columns(2)
    col2.selectbox("Select Search Phrase for Bloom:", myBloomlist,key="phrase_selection")

    "---"
    submitted_bloom_selection = st.form_submit_button("Display Bloom Result")
    if submitted_bloom_selection:
        actual_phrase = str(st.session_state["phrase_selection"]).replace(" ","%20")
        iframe_src =f"""https://bloom.neo4j.io/index.html?connectURL=0a10ec94.databases.neo4j.io&run=true&search={actual_phrase}"""
        iframe_src2 ="https://bloom.neo4j.io/index.html?connectURL=c2f7f38c.databases.neo4j.io&run=true&search=clear"
        components.iframe(iframe_src, height=800, scrolling=True)


