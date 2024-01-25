import os

import streamlit as st
from langchain import hub  #new
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)
from email import process_email

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

def main():

    st.header("📄You Got Mail!")

    # Create a text input box and a button
    run_button = st.button("Run", key="Run")
    query = st.text_input("Query", key="Query")

    # Initialize containers for the output
    response_container = StreamHandler(st.empty())

    # Logic to update output areas when the 'Run' button is clicked
    if run_button:
        vectorstore = process_email(
            imap_server, 
            email_account, imap_password, 
            address_book, 
            days_to_search
        )
        chain = configure_email_qa_rag_chain(llm, vectorstore)

    if query:
        result = ask_question(question, chain)

if __name__ == "__main__":
    main()
