from collections import defaultdict
import json
import textwrap
import chromadb
import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


st.set_page_config(layout="wide")


template = """Use the following pieces of context from congressional legislation to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

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

#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
#llm = ChatOpenAI(model_name="gpt-4", temperature=0)



st.title("Q&A Over Congressional Legislation")


def get_sponsor_link(sponsors):
    base_url = "https://bioguide.congress.gov/search/bio"
    dd = json.loads(sponsors)[0]
    url = "{}/{}".format(base_url, dd["bioguideId"])
    return "[{}]({})".format(dd["fullName"], url)

col1, col2 = st.columns(2)

with col1:

    with st.form("my_form"):
        query = st.text_area('Enter question:')
        n_ret_docs = st.slider(
            'N retrieved chunks',
            min_value=1,
            max_value=20,
            value=10,
            step=1,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": n_ret_docs}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True,
        )
        submitted = st.form_submit_button('Submit')

    if submitted:
        out = qa_chain({"query": query})
        st.info(out["result"])

def wrap_text(text):
    return "\n".join(textwrap.wrap(text))


with col2:

    if submitted:

        # group source documents
        grpd_source_docs = defaultdict(list)
        for doc in out["source_documents"]:
            grpd_source_docs[doc.metadata["parent_id"]].append(doc)

        # sort chunks in each group by start index
        for key in grpd_source_docs:
            grpd_source_docs[key] = sorted(
                grpd_source_docs[key],
                key=lambda x: x.metadata["start_index"],
            )

        # sort groups by number of chunks
        grpd_source_docs = sorted(
            tuple(grpd_source_docs.items()),
            key=lambda x: -len(x[1]),
        )

        for parent_id, doc_grp in grpd_source_docs:
            first_doc = doc_grp[0]
            ref = "{} chunks from {}\n\n[{}]({})\n\n{}".format(
                len(doc_grp),
                first_doc.metadata["parent_id"],
                first_doc.metadata["title"],
                first_doc.metadata["congress_gov_url"],
                get_sponsor_link(first_doc.metadata["sponsors"]),
            )
            doc_contents = [
                wrap_text(
                    "[start_index={}] ".format(doc.metadata["start_index"]) +
                    doc.page_content
                ) for doc in doc_grp
            ]
            with st.expander(ref):
                st.text("\n\n...\n\n".join(doc_contents))




