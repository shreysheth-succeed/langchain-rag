import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddingModel = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

FILE_PATH = "data\\Sensors.pdf"
FILENAME = os.path.basename(FILE_PATH).split(".pdf")[0]
CHROMA_PATH = f"vector_store\\{FILENAME}"

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents: list[Document]):
    text_splitter = SemanticChunker(embeddingModel, breakpoint_threshold_type='percentile')
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: list[Document], chroma_path: str):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    Chroma.from_documents(
        chunks, embeddingModel, persist_directory=chroma_path
    )
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")

def generate_data_store():
    documents = load_documents(FILE_PATH)
    chunks = create_chunks(documents)
    save_to_chroma(chunks, CHROMA_PATH)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()