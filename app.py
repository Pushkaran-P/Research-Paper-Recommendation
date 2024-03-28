# Imports

import pandas as pd
import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# Loading dataframe and combining title and abstract

df = pd.read_csv('database.csv')

df['abstract'] = df['abstract'].fillna(' No Abstract Available ')
df['combined'] = 'The title of this research paper is ' + df['title'] + '. The abstract of this research paper: ' + df['abstract']
df['Id'] = df['Id'].astype(str)

# ChromaDB expects a list of documents
documents = df["combined"].tolist()
# ChromaDB expects a List of dicitonaries for its metadata
metadatas = [{ 'year': row['year'], 'authors': row['authors'], 'citations': row['citations'], 'original_id': row['Id'] } for _, row in df.iterrows()]
# ChromaDB expects a list of ids for its ids
ids = df["Id"].tolist()

# Initializing the Huggingface embeddings here

model_name = "BAAI/bge-base-en-v1.5"
# Device can be changed to "cuda:0" or "cpu"
model_kwargs = {"device": "cuda:0"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Creating Chroma db here
db = Chroma.from_texts(texts = documents, collection_name = 'research_db',
                       embedding = embeddings, persist_directory = "/content/research_db",
                       metadatas = metadatas,
                       ids = ids)

import os
import zipfile

def zip_folder(folder_path, zip_name):
    # Create a ZipFile object
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over all the files in the folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Write each file into the zip file
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(folder_path, '..')))

# Use the function to zip the folder
zip_folder('/content/research_db', 'research_db.zip')

# Initialize retriever (very imp)
retriever = db.as_retriever()

import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Download and use the llm engine

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map = "auto")
llm = HuggingFacePipeline(pipeline=generate_text)

# RunnablePassthrough to get user input
# StrOutputParser to parse output
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Contextualize prompts help in retrieving via chat history but other methods can be used
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)
import streamlit as st

# @st.cache(allow_output_mutation=True)
# def load_model():
#     # Initialize your model here
#     rag_chain = (
#         RunnablePassthrough.assign(
#             context=contextualized_question | retriever | format_docs
#         )
#         | qa_prompt
#         | llm
#     )
#     return rag_chain
# rag_chain = load_model()


from langchain_core.messages import AIMessage, HumanMessage
import sys 

def chatbot(question):
    if question.lower() == 'exit':
        st.write('Exiting')
        sys.exit()
    if question == '':
        return
    ai_msg = rag_chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=question), ai_msg])
    return ai_msg

st.title('Research Paper Chatbot')

# Initialize the session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display the entire conversation history
for chat in st.session_state.chat_history:
    st.markdown(f"**User**: {chat['user']}")
    st.markdown(f"**Bot**: {chat['bot']}")

prev_qry = ""

user_input = st.text_input("Input: ")

# if user_input:
#     response = chatbot(user_input)
#     st.session_state.chat_history.append({"user": user_input, "bot": response})

if prev_qry != user_input:
    prev_qry = user_input
    response = chatbot(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})