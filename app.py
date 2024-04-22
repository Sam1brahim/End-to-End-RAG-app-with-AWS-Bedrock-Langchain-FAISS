import json
import os
import sys
import boto3
import streamlit as st
#--------------------------------------------------------------

## Using the embeddings model to generate embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
#--------------------------------------------------------------

## Data Ingestion (loading my docs)

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000) # to fit into the context window of my model
    docs=text_splitter.split_documents(documents)
    return docs
#--------------------------------------------------------------
## Conversion into Vector embeddings, and setting a remote Vector Store
from langchain_community.vectorstores import FAISS
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ## getting Anthropic Model
    llm= Bedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                 model_kwargs={"max_tokens":512}) # invoke the wrapper of langchain to connect to AWS bedrock
    return llm
def get_mixtral_llm():
    ## getting Anthropic Model
    llm= Bedrock(model_id="mistral.mixtral-8x7b-instruct-v0:1",
                 model_kwargs={"max_tokens":512}) # invoke the wrapper of langchain to connect to AWS bedrock
    return llm
#--------------------------------------------------------------

## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )

    answer = qa({"query":query})
    return answer['result']
#--------------------------------------------------------------
## Calling Bedrock Client

bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings=BedrockEmbeddings(model_id="cohere.embed-english-v3",client=bedrock)


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock 🍕")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Mixtral 8x7b Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_mixtral_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
