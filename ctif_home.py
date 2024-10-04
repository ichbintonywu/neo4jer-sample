import streamlit as st
from st_pages import Page, show_pages, add_page_title
add_page_title()
show_pages(
    [
        Page("_pages/ctif_shareEmailActivity.py", "Shared Email Analysis", ":email:"),
        Page("_pages/ctif_erActivity.py", "ER Activity Analysis", ":cop:"),    
        Page("_pages/ctif_bloom.py", "Aura Bloom with Fixed Phrase", ":blossom:"),
        Page("_pages/ctif_bloom_sharedAttr.py", "Aura Bloom ER Shared Profile", ":blossom:"),
        Page("_pages/ctif_chat.py", "GraphRAG Bot", ":robot_face:"),
        ]
    )
