import os
os.environ['TRANSFORMERS_CACHE'] = '/datadrive/hugging-face-cache'

import streamlit as st

# check gpu
from torch import cuda
# used to log into huggingface hub
from huggingface_hub import login
# used to setup language generation pipeline
import torch

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.prompts.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import LangchainEmbedding, VectorStoreIndex, ServiceContext
from llama_index import SimpleDirectoryReader

st.title('【ＳＡＳ】 Quickstart App')

MODEL = "meta-llama/Llama-2-7b-chat-hf"

# system_prompt = """
# You are an assistant and your job is to only answer questions about SAS Viya 4, based on the given source documents. Here are some rules you must always follow:
# - Assume if the question is asking about SAS Viya, it is referring to SAS Viya 4.
# - If a question is not related to SAS Viya 4, respond that you only answer questions related to SAS Viya 4.
# - Keep your answers based on facts, be as descriptive as possible when describing SAS Viya functionality and do not hallucinate features.
# - When answering questions, please use dot points and bulleted lists in your answer when detailing a list of instructions.
# - Never say thank you or that you are an AI agent, just answer the question directly.
# - Stop being polite and just be direct.
# """
system_prompt = """"""
# This will wrap the default prompts that are internal to llama-index
# query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + system_prompt + "<</SYS>>\n\n{query_str}[/INST] "
)

@st.cache_resource
def load_documents2():
    return SimpleDirectoryReader(
        input_dir="doc_data/", 
        exclude=["*.md"]
        ).load_data()

@st.cache_resource
def embeddings():
    return LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

@st.cache_resource
def load_model2():
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={
            "temperature": 0.1,
             "repetition_penalty": 1.1},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float32}
    )

llm = load_model2()
embed_model = embeddings()
documents = load_documents2()
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
# query_engine = index.as_query_engine()


def generate_response2(input_text):
    return st.info(chat_engine.chat(input_text)) #st.info(query_engine.query(input_text)) 

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are some common customizations I can add to my SAS Viya deployment?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        print("Response submitted")
        generate_response2(text)