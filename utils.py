import pandas as pd
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate


def make_FAISS_db(embedding_model_name):
    """make_FAISS_db generates vector database for storing the embeddings.

    Arguments:
        embedding_model_name {str} -- the model to use for creating embeddings

    Returns:
        FAISS -- the database
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    df = pd.read_csv("articles.csv")
    articles = DataFrameLoader(df, page_content_column="Text")
    documents = articles.load()
    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_texts = splitter.split_documents(documents)
    print("FAISS database is generating...")
    FAISS_database = FAISS.from_documents(splitted_texts, embedding_model)
    return FAISS_database


def prompt_template():
    """prompt_template generates prompt template which will be filled out with context form RAG and question from user.
    The template is standard, but with "I don't know" option, which is essential for generating science text.

    Returns:
        HuggingFacePipeline -- ready pipeline for text generation
    """
    prompt_template = """
        [INST]
        Answer the question based on the context below. 
        If you do not know the answer, or are unsure, say you don't know..

        Context:
        {% for doc in context %}
            {{ doc.page_content }}
        {% endfor %}
        [/INST]

        Question:
        {{question}}
        [/INST]"""
    return PromptTemplate(
        input_variables=["context", "question"], template=prompt_template, template_format="jinja2"
    )
