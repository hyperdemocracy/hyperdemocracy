from collections import defaultdict
import json
from pathlib import Path
import textwrap

import chromadb
import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


st.set_page_config(layout="wide")


GGUF_PATH = Path("/home/galtay/data/gguf_models")


template = """Use the following pieces of context from congressional legislation to answer the question at the end.
Remember that you can only answer questions about the content of the legislation. If you don't know the answer, just say that you don't know, don't try to make up an answer.
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


st.title("LegisQA: Explore the Legislation of the 118th Congress")
st.write(
    """Pose inquiries regarding legislation authored by the 118th Congress in 2023
Choose a model, pose an inquiry, and see a response.
This application can be used to compare a number of private and open-source language models, using as a common basis the primary sources of bills proposed by the US Congress. some models require longer to run than others.
Please see hyperdemocracy.us and join our upcoming sessions, where you can learn to code using LLMs and legislation, and exchange ideas on how democracy should change, looking ahead in this young millenium."""
)


LLM_PROVIDERS_MODELS = {
    "openai-chat": [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
    ],
    "hfhub": [
        "databricks/dolly-v2-3b",
        "google/flan-t5-large",
        "google/flan-t5-small",
    ],
    "llamacpp": sorted(list(GGUF_PATH.glob("*.gguf"))),
}


def get_sponsor_link(sponsors):
    base_url = "https://bioguide.congress.gov/search/bio"
    dd = json.loads(sponsors)[0]
    url = "{}/{}".format(base_url, dd["bioguideId"])
    return "[{}]({})".format(dd["fullName"], url)


with st.sidebar:

   n_ret_docs = st.slider(
       'N retrieved chunks',
       min_value=1,
       max_value=40,
       value=10,
       step=1,
   )

   llm_provider = st.selectbox(
       label="llm provider",
       options=LLM_PROVIDERS_MODELS.keys(),
   )

   llm_name = st.selectbox(
       label="llm",
       options=LLM_PROVIDERS_MODELS[llm_provider],
   )

   if llm_provider == "openai-chat":
       temperature = st.slider(
           'temperature',
           min_value=0.0,
           max_value=2.0,
           value=0.0,
           step=0.05
       )
       top_p = st.slider(
           'top_p',
           min_value=0.0,
           max_value=1.0,
           value=1.0,
           step=0.05
       )

   if llm_provider == "llamacpp":
       n_ctx = st.slider(
           'n_ctx',
           min_value=512,
           max_value=16384,
           value=4096,
           step=64
       )
       max_tokens = st.slider(
           'max_tokens',
           min_value=512,
           max_value=16384,
           value=512,
           step=64
       )
       temperature = st.slider(
           'temperature',
           min_value=0.0,
           max_value=1.5,
           value=0.0,
           step=0.05
       )
       top_p = st.slider(
           'top_p',
           min_value=0.0,
           max_value=1.0,
           value=1.0,
           step=0.05
       )


col1, col2 = st.columns(2)

with col1:

    with st.form("my_form"):

        query = st.text_area('Enter question:')

        if llm_provider == "openai-chat":
            llm = ChatOpenAI(model_name=llm_name, temperature=0)

        elif llm_provider == "hfhub":
            llm = HuggingFaceHub(
                repo_id=llm_name, model_kwargs={"temperature": 0.0, "max_length": 64}
            )

        elif llm_provider == "llamacpp":
            llm = LlamaCpp(
                model_path="/home/galtay/data/gguf_models/llama2-7b-chat-Q4_K_M.gguf",
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n_ctx=n_ctx,
            )
        else:
            st.error(f"{llm_name=} not recognized")
            st.stop()

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": n_ret_docs}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True,
        )
        submitted = st.form_submit_button('Submit')


def escape_markdown(text):
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!$"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\"+char)
    return text

if submitted:
    out = qa_chain({"query": query})
    with col1:
        st.info(escape_markdown(out["result"]))
else:
    st.stop()


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
                "[start_index={}] ".format(doc.metadata["start_index"]) + doc.page_content
                for doc in doc_grp
            ]
            with st.expander(ref):
                st.write(
                    escape_markdown("\n\n...\n\n".join(doc_contents))
                )




