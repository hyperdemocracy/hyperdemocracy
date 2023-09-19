from collections import defaultdict
import datetime
import json
import os
from pathlib import Path

import chromadb
import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import openai


st.set_page_config(layout="wide")


env_openai_api_key = os.getenv("OPENAI_API_KEY")
env_hfhub_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
env_gguf_path = os.getenv("GGUF_PATH")


LLM_PROVIDERS = ["openai-chat", "hfhub", "llamacpp"]
OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
]



def load_vectorstore():
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
    return vectorstore


def get_sponsor_link(sponsors):
    base_url = "https://bioguide.congress.gov/search/bio"
    dd = json.loads(sponsors)[0]
    url = "{}/{}".format(base_url, dd["bioguideId"])
    return "[{}]({})".format(dd["fullName"], url)


vectorstore = load_vectorstore()


DEFAULT_PROMPT_TEMPLATE = """Use the following pieces of context from congressional legislation to answer the question at the end.
Remember that you can only answer questions about the content of the legislation. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""



st.title(":classical_building: LegisQA :computer:")
st.header("Explore the Legislation of the 118th US Congress")
st.write(
    """When you send a question to LegisQA, it will attempt to retrieve relevant content from the [118th United States Congress](https://en.wikipedia.org/wiki/118th_United_States_Congress), pass it to a [large language model (LLM)](https://en.wikipedia.org/wiki/Large_language_model), and generate an answer. This technique is known as Retrieval Augmented Generation (RAG). You can read the [original paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) or a [recent summary](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) to get more details. Once the answer is generated, the retrieved content will be available for inspection with links to the bills and sponsors.
This technique helps to ground the LLM response by providing context from a trusted source, but it does not guarantee a high quality answer. We encourage you to play around. Try different models. Find questions that work and find questions that fail."""
)




with st.sidebar:

    st.subheader(":brain: Learn about [hyperdemocracy](https://hyperdemocracy.us)")
    st.subheader(":world_map: Visualize with [nomic atlas](https://atlas.nomic.ai/map/b65c568b-ce37-40a1-b376-b20cc7580118/91b3337c-f6c9-4c1c-a755-18c84aa4141c)")

    st.divider()

    llm_provider = st.selectbox(
        label="llm provider",
        options=LLM_PROVIDERS,
    )

    if llm_provider == "openai-chat":
        llm_name_options = OPENAI_CHAT_MODELS
        llm_name = st.selectbox(
            label="llm",
            options=llm_name_options,
        )
        if env_openai_api_key is None:
            openai_api_key = st.text_input(
                "Provide your OpenAI API key here (sk-...)",
                type="password",
            )
        else:
            openai_api_key = env_openai_api_key
        if openai_api_key == "":
            st.stop()


    elif llm_provider == "hfhub":
        llm_name = st.text_input(
            "Provide a HF model name (google/flan-t5-large)",
        )
        if env_hfhub_api_key is None:
            hfhub_api_token = st.text_input(
                "Provide your HF API token here (hf_...)",
                type="password",
            )
        else:
            hfhub_api_token = env_hfhub_api_key
        if hfhub_api_token == "" or llm_name == "":
            st.stop()


    elif llm_provider == "llamacpp":
        if env_gguf_path is None:
            gguf_path = st.text_input("Provide a path to *.gguf files")
        else:
            gguf_path = env_gguf_path
        if gguf_path == "":
            st.stop()
        else:
            gguf_path = Path(gguf_path)
        if not gguf_path.exists():
            st.warning("Provided gguf path does not exists")
            st.stop()
        gguf_files = sorted(list(gguf_path.glob("*.gguf")))
        if len(gguf_files) == 0:
            st.warning("Provided gguf path contains no gguf files")
            st.stop()
        gguf_map = {gf.name: gf for gf in gguf_files}
        llm_name_options = gguf_map.keys()
        llm_name = st.selectbox(
            label="llm",
            options=llm_name_options,
        )

    with st.expander("Retrieval parameters"):

        n_ret_docs = st.slider(
            'Number of chunks to retrieve',
            min_value=1,
            max_value=40,
            value=10,
        )

    with st.expander("Generative parameters"):

        temperature = st.slider('temperature', min_value=0.0, max_value=2.0, value=0.0)
        top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=1.0)

        if llm_provider == "openai-chat":
            llm = ChatOpenAI(
                model_name=llm_name,
                temperature=temperature,
                openai_api_key=openai_api_key,
            )

        elif llm_provider == "hfhub":
            max_length = st.slider(
                'max_tokens',
                min_value=512,
                max_value=16384,
                value=512,
                step=64
            )
            llm = HuggingFaceHub(
                repo_id=llm_name,
                huggingfacehub_api_token=hfhub_api_token,
                model_kwargs={
                    "temperature": temperature,
                    "max_length": max_length,
                }
            )

        elif llm_provider == "llamacpp":
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
            llm = LlamaCpp(
                model_path=str(gguf_map[llm_name]),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n_ctx=n_ctx,
            )


    with st.expander("Prompt"):
        prompt_template = st.text_area(
            "prompt template",
            DEFAULT_PROMPT_TEMPLATE,
            height=300,
        )


qa_chain_prompt = PromptTemplate.from_template(prompt_template)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": n_ret_docs}),
    chain_type_kwargs={"prompt": qa_chain_prompt},
    return_source_documents=True,
)


col1, col2 = st.columns(2)

with col1:

    with st.form("my_form"):
        query = st.text_area('Enter question:')
        submitted = st.form_submit_button('Submit')


def escape_markdown(text):
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!$"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\"+char)
    return text

if submitted:
    out = qa_chain({"query": query})
    st.session_state["out"] = out


if not "out" in st.session_state:
    st.stop()
else:
    out = st.session_state["out"]



with col1:
    do_escape_markdown = st.checkbox("escape markdown in answer")
    if do_escape_markdown:
        st.info(escape_markdown(out["result"]))
    else:
        st.info(out["result"])

with col2:

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
        ref = "{} chunks from {}\n\n{}\n\n[congress.gov]({}) | [govtrack.us]({})\n\n{}".format(
            len(doc_grp),
            first_doc.metadata["parent_id"],
            first_doc.metadata["title"],
            first_doc.metadata["congress_gov_url"],
            first_doc.metadata["govtrack_url"],
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

with st.sidebar:

    download_pack = {
        "llm_provider": llm_provider,
        "llm_name": llm_name,
        "query": query,
        "prompt_template": prompt_template,
        "out_result": out["result"],
        "out_source_documents": [doc.dict() for doc in out["source_documents"]],
        "time": datetime.datetime.now().isoformat(),
    }
    st.download_button(
        label="Download Results",
        data = json.dumps(download_pack, indent=4),
        file_name='test.json',
        mime='text/json',
    )
