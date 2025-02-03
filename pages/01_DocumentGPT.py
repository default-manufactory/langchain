import time
from uuid import UUID
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.document_loaders import UnstructuredFileLoader ,PyPDFLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chat_models.openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    streaming=True,
    model="gpt-4o",
    temperature=0.0,
    callbacks=[
        ChatCallbackHandler(),
    ]
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    

    loader = PyPDFLoader(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better relevance
        chunk_overlap=300,  # Sufficient overlap for context continuity
        separators=["\n\n", "\n", " ", ""],
    )
    docs = loader.load_and_split(text_splitter=splitter)
    
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If the context is insufficient I am pretty sure there is more infomation so look for it deeply. Look through more files if you don't have information.
            Provide concise and accurate answers. Avoid verbosity and unrelated information. make the answer 500 to 1500 words. Also, I want you to separate paragraphs.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []