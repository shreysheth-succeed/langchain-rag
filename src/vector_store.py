import nltk
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from src.ingestion import process_pdf

# Download the sentence tokenizer
nltk.download("punkt")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def semantic_chunking(text, similarity_threshold=0.8):
    """Splits text into semantically coherent chunks based on embedding similarity."""
    sentences = nltk.sent_tokenize(text)
    sentence_embeddings = np.array(embedding_model.embed_documents(sentences))
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i - 1]) / (
            np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[i - 1])
        )
        
        if similarity < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def store_in_vector_db(chunks):
    """Stores text chunks in ChromaDB."""
    vector_db = Chroma(collection_name="pdf_chunks", embedding_function=embedding_model)
    vector_db.add_texts(chunks)
    return vector_db

def process_and_store(pdf_path):
    """Extracts text, chunks semantically, and stores embeddings."""
    text, tables = process_pdf(pdf_path)
    chunks = semantic_chunking(text + "\n".join(tables))
    return store_in_vector_db(chunks)
