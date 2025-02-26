import os
from dotenv import load_dotenv
from argparse import ArgumentParser
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "vector_store/Sensors"



PROMPT_TEMPLATE = """
Answer the question based only on the following context. If answer not found within the context, tell user that "I am sorry, I can't help with that :("

Context:
{context}

---

Question: {question}
"""

def main():
    # Create CLI.
    parser = ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    # embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    normalized_results = [(doc, (1 + score) / 2) for doc, score in results]
    # filtered_results = [doc for doc in normalized_results if doc[1] >= 0.7]

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in normalized_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatOpenAI(api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in normalized_results]
    formatted_response = f"Response:\n{response_text.content}\n\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()