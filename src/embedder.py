from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import os

DATA_PATH = "./data/"
DB_PATH = "./vectorstores/db"


def create_vector_db():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Processed {len(documents)} pdf pages!")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    ollama_embeddings = OllamaEmbeddings(model="llama2")

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=ollama_embeddings,
        persist_directory=DB_PATH,
    )
    vectorstore.persist()
    print("Finished Adding the Chunks to the Vector Database")


if __name__ == "__main__":
    create_vector_db()
