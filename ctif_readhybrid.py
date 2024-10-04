import os
from neo4j import GraphDatabase
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import  RunnableParallel, RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings
import json
import streamlit as st
import ast
from openai import AzureOpenAI

NEO4J_HOST = st.secrets["NEO4J_AURA_CTIF"]+":"+st.secrets["NEO4J_PORT"] 
NEO4J_USER = st.secrets["NEO4J_AURA_CTIF_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_AURA_CTIF_PASSWORD"]
NEO4J_DATABASE = 'neo4j'
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]

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
deployment_name = AZURE_DEPLOYMENT_NAME

def get_embedding(client, text, model):
    response = client.embeddings.create(
                    input=text,
                    model=model,
                )
    return response.data[0].embedding

driver = GraphDatabase.driver(NEO4J_HOST, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)

openai_client = AzureOpenAI(
    api_key = os.environ["AZURE_OPENAI_API_KEY"] ,
    api_version = os.environ["OPENAI_API_VERSION"],
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"] 
)
deployment_name = AZURE_DEPLOYMENT_NAME

def query(user_question):

    # cypher query
    retrieval_query = """
        WITH node AS chunk, score 
        MATCH (chunk)--(d:Document)
        OPTIONAL MATCH (c1:Chunk)-[:NEXT_CHUNK]->(chunk)
        OPTIONAL MATCH (chunk)-[:NEXT_CHUNK]->(c2:Chunk)
        WITH d, c1 as pre, c2 as next,chunk,chunk.position as block_idx,score
        WITH d, apoc.coll.flatten([collect(chunk.text),collect(pre.text),collect(next.text)]) AS a1, collect(chunk)+COLLECT(pre)+COLLECT(next) as chunks,score
        WITH d, reduce(text = "", x IN a1 | text + x + '.') AS text,chunks, max(score) AS maxScore
        RETURN {source: d.fileName,page: d.page_number} AS metadata, 
            text, maxScore AS score LIMIT 1;
    """

    vector_index = Neo4jVector.from_existing_index(
        embedding = AzureOpenAIEmbeddings(),
        search_type="vector",
        index_name = "vector",
        embedding_node_property="value",
        url = NEO4J_HOST,
        username = NEO4J_USER,
        password = NEO4J_PASSWORD,
        database = NEO4J_DATABASE,
        retrieval_query = retrieval_query
    )

    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the person, organization,or business entities that "
            "appear in the text",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)

    # revoke_invoke = entity_chain.invoke({"question": user_question})
    # revoke_str_list = str(revoke_invoke)[6:]
    # revoke_invoke_entity_sentence = ast.literal_eval(revoke_str_list)
    # my_reembedding = get_embedding(openai_client, revoke_invoke_entity_sentence,EMBEDDING_MODEL)
    # print (my_reembedding)
    # result = " ".join(revoke_invoke_entity_sentence)
    
    # driver = GraphDatabase.driver(NEO4J_HOST, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)

    # with driver.session() as session:
    #     retrieval_query = f"""
    #         WITH {my_reembedding} AS e
    #         CALL db.index.vector.queryNodes('vector',3, e) yield node, score
    #         with node as Comp, score 
    #         WITH node AS chunk, score 
    #     MATCH (chunk)--(d:Document)
    #     OPTIONAL MATCH (c1:Chunk)-[:NEXT_CHUNK]->(chunk)
    #     OPTIONAL MATCH (chunk)-[:NEXT_CHUNK]->(c2:Chunk)
    #     WITH d, c1 as pre, c2 as next,chunk,chunk.position as block_idx,score
    #     WITH d, apoc.coll.flatten([collect(chunk.text),collect(pre.text),collect(next.text)]) AS a1, collect(chunk)+COLLECT(pre)+COLLECT(next) as chunks,score
    #     WITH d, a1 AS answers,chunks, max(score) AS maxScore
    #     RETURN {{source: d.fileName}} AS metadata, 
    #         answers AS text, maxScore AS score LIMIT 3
    #         """
    #     result = session.run(
    #         retrieval_query
    #         )
    #     count = 0
    #     all_text_info = ""
    #     for record in result:
    #         all_text_info = all_text_info + str(record["text"])
    #         # score = record["score"]
    #         print (all_text_info)
    #         count = count + 1
    #     session.close()

    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Fulltext index query
    # def structured_retriever(question: str) -> str:
    #     """
    #     Collects the neighborhood of entities mentioned
    #     in the question
    #     """
    #     result = ""
    #     entities = entity_chain.invoke({"question": question})
    #     print("Entities detected: ", entities)
    #     for entity in entities.names:
    #         response = graph.query(
    #             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
    #             YIELD node,score
    #             CALL {
    #             MATCH (node)-[r:!MENTIONS]->(neighbor)
    #             RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
    #             UNION
    #             MATCH (node)<-[r:!MENTIONS]-(neighbor)
    #             RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
    #             }
    #             RETURN output LIMIT 50
    #             """,
    #             {"query": generate_full_text_query(entity)},
    #         )
    #         print("Full Text Query: ", generate_full_text_query(entity))
    #         result += "\n".join([el['output'] for el in response])
    #     print("Full Text Query Results: ", result)
    #     return result

    def retriever(question: str):
        print(f"Search query: {question}")
        # structured_data = structured_retriever(question)
        unstructured_data = vector_index.similarity_search(question)
        print(unstructured_data)
        # unstructured_page_content = [el.page_content for el in unstructured_data]
        # unstructured_metadata = [(json.dumps(el.metadata)) for el in unstructured_data]
        # print("Unstructured Data: ", unstructured_page_content)
        # print("Metadata: ", unstructured_metadata)
        # # final_data = f"""Structured data:
        # #                 {structured_data}
        # #                 Unstructured data:
        # #                 {"#Document ". join(unstructured_data)}
        # #             """
        # final_data = f"""Unstructured data:
        #                 {"#Document ". join(unstructured_page_content)}
        #                 Metadata:
        #                 {"#Metadata ". join(unstructured_metadata)}
        #             """
        return unstructured_data


    # Condense a chat history and follow-up question into a standalone question
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | AzureChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
    )

    template = """Answer the question based only on the following context:
    {context}
    Only use information from the context. You're help to summarize,please rephrase violence words, and nouns like hate, make the answer as comprehensive as possible.
    Question: {question}
    Use natural language to answer the question.
    Answer:
    #----
    Answer_Body
    #----
    At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    LIST DOWN ALL THE PAGES
    For example, if context has `metadata`:(source:'docu_url', page:1), you should display:
    Metadata:
    - Document: 'doc_url', Page: 1 
    - Document: 'doc_url', Page: x
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"question": user_question})
    
    return answer
