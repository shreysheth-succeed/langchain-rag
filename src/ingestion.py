import os
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

FILE_PATH = "data\\Sensors.pdf"
FILENAME = os.path.basename(FILE_PATH).split(".pdf")[0]
CHROMA_PATH = f"vector_store\\{FILENAME}"

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents: list[Document]):
    text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), breakpoint_threshold_type='percentile')
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: list[Document], chroma_path: str):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), persist_directory=chroma_path
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")

def generate_data_store():
    documents = load_documents(FILE_PATH)
    chunks = create_chunks(documents)
    save_to_chroma(chunks, CHROMA_PATH)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()