from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import TextLoader
import os

doc_folder = '/Users/<PATH>'
chat_history = []


def list_files(path):
    # Get all files in the directory
    files = os.listdir(path)

    # Loop through the files and print their names
    return files


def load_docs(document_folder, docs=[]):
    file_names = list_files(document_folder)
    for file_name in file_names:
        full_path = os.path.join(doc_folder, file_name)
        loader = TextLoader(full_path)
        docs.extend(loader.load())
    return docs


def split_docs_into_chunks_from(document_folder, chunk_size, overlap=30):
    docs = load_docs(document_folder)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=overlap)
    documents = text_splitter.split_documents(docs)
    return documents


def create_vector_db_from_docs(document_folder, chunk_size):
    documents = split_docs_into_chunks_from(document_folder, chunk_size, 30)
    embeddings = OpenAIEmbeddings()
    dir = "persist_seed"
    vector_store = Chroma.from_documents(
        documents, persist_directory=dir)
    vector_store.persist()

    # Now we can load the persisted database from disk, and use it as normal.

    vector_store = Chroma(persist_directory=dir)
    return vector_store


def prep_chat_vector_chain(llm, chain_type="map_rerank"):
    doc_chain = load_qa_chain(llm, chain_type=chain_type)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    return question_generator, doc_chain


def create_chat_vector_chain(vector_store, llm):
    question_generator, doc_chain = prep_chat_vector_chain(llm)
    chain = ChatVectorDBChain(
        vectorstore=vector_store,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )
    return chain


def ask_chat_vector_chain(query, chain, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    return result["answer"]


def get_answer(query, llm, document_folder, chat_history=[], chunk_size=200):
    vector_store = create_vector_db_from_docs(
        document_folder, chunk_size=chunk_size)
    chain = create_chat_vector_chain(vector_store, llm)
    return ask_chat_vector_chain(query, chain, chat_history)


def get_answer_from_vector(query, llm, vector_store, chat_history=[]):
    chain = create_chat_vector_chain(vector_store, llm)
    return ask_chat_vector_chain(query, chain, chat_history)

chat_history = []
temp = 0
llm = OpenAI(temperature=temp)
vector_store = create_vector_db_from_docs(doc_folder, 2000)

questions = ["what is"]
for q in questions:
    ans = get_answer_from_vector(q, llm, vector_store, chat_history)
    print(ans)
    chat_history.append(ans)

