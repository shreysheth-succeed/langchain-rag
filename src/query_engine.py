import os
from argparse import ArgumentParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ["ANTHROPIC_API_KEY"] = "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"
chat_model = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)


# Define a prompt template for generating a refined prompt
prompt_generator_template = PromptTemplate(
    input_variables=["question", "context"],
    template="Given the context: {context}, generate a well-structured prompt for an AI to answer the question: {question}"
)

# Define a prompt template for answering the generated prompt
answer_template = PromptTemplate(
    input_variables=["generated_prompt"],
    template="You are an AI assistant. Please provide a detailed response to the following prompt: {generated_prompt}"
)

# Create a sequential chain
full_chain = ({"question": RunnablePassthrough(), "context": RunnableLambda(lambda _: "The water level in Himalaya has risen by 5000 meters above sea level")} | prompt_generator_template | chat_model | answer_template | chat_model | StrOutputParser())
response = full_chain.invoke("what is the weather in ahmedabad today?")
print(response)