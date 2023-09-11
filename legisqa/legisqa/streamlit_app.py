import chromadb
import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


template = """Use the following pieces of context from congressional legislation to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

n_ret_docs = 10

model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedder = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


chroma_client = chromadb.PersistentClient(path="chroma.db")
collection = chroma_client.get_collection(
    name='uscb',
    embedding_function=embedder.embed_documents,
)
vectorstore = Chroma(
    collection_name="uscb",
    embedding_function=embedder,
    persist_directory="chroma.db",
    client=chroma_client,
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": n_ret_docs}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
)


st.title("Q&A Over Congressional Legislation")


def answer(query):
    out = qa_chain({"query": query})
    refs = [
        "[{}]({})".format(
            doc.metadata["title"],
            doc.metadata["congress_gov_url"]
        ) for doc in out["source_documents"]
    ]

    txt = ""
    txt += out["result"]
    txt += "\n\nRetrieved Sources:\n\n"
    txt += "\n\n".join(refs)

    st.info(txt)


with st.form("my_form"):
    query = st.text_area('Enter question:')
    submitted = st.form_submit_button('Submit')
    if submitted:
        answer(query)
