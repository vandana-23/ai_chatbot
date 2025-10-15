import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "vector_db")
Path(DB_DIR).mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "google/flan-t5-large"       
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2" 

def _choose_model_name():
    return DEFAULT_MODEL 

def load_retriever(persist_directory: str = DB_DIR, k: int = 5):
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.4, "k": k}
    )
    return retriever


def load_llm(model_name: str = None, max_new_tokens: int = 256):
    model_name = model_name or _choose_model_name()

    ## Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )

    ## Text2text generation pipeline
    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=gen)

def get_prompt_template():
    return PromptTemplate.from_template(
        """
        You are an intelligent assistant for Goklyn Company.
        Use ONLY the context below to answer accurately and concisely.
        If the context does not contain the answer, say:
        "I don't have enough information to answer that right now."

        Context:
        {context}

        Question:
        {question}

        Instructions:
        - Do not make up facts.
        - Use bullet points if listing items.
        - Mention the source if possible.

        Final Answer:
        """
    )


def build_qa_chain(model_name: str = None, persist_directory: str = DB_DIR, k: int = 3):  
    retriever = load_retriever(persist_directory=persist_directory, k=k)
    llm = load_llm(model_name)
    prompt = get_prompt_template()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa

_QA_CHAIN = None

def get_qa_chain(model_name: str = None, persist_directory: str = DB_DIR, k: int = 3):
    global _QA_CHAIN
    if _QA_CHAIN is None:
        _QA_CHAIN = build_qa_chain(
            model_name=model_name,
            persist_directory=persist_directory,
            k=k
        )
    return _QA_CHAIN
