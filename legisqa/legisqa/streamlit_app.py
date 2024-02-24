from collections import defaultdict
import datetime
import json
import os
from pathlib import Path

import chromadb
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import openai


st.set_page_config(layout="wide", page_title="LegisQA")


env_openai_api_key = os.getenv("LEGISQA_OPENAI_API_KEY")
env_hfhub_api_key = os.getenv("LEGISQA_HUGGINGFACEHUB_API_TOKEN")
env_gguf_path = os.getenv("LEGISQA_GGUF_PATH")
env_chroma_path = os.getenv("LEGISQA_CHROMA_PATH")


CONGRESS_GOV_TYPE_MAP = {
    "hconres": "house-concurrent-resolution",
    "hjres": "house-joint-resolution",
    "hr": "house-bill",
    "hres": "house-resolution",
    "s": "senate-bill",
    "sconres": "senate-concurrent-resolution",
    "sjres": "senate-joint-resolution",
    "sres": "senate-resolution",
}


LLM_PROVIDERS = ["openai-chat", "hfhub", "llamacpp"]
OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
]


def load_vectorstore():
    model_name = "BAAI/bge-small-en-v1.5"
    collection_name = "usc-113-to-118-vecs-v1-s1024-o256-BAAI-bge-small-en-v1.5"

    chroma_client = chromadb.PersistentClient(path=env_chroma_path)
    collection = chroma_client.get_collection(name=collection_name)
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    emb_fn = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this question for searching relevant passages: ",
    )
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=emb_fn,
    )
    return vectorstore


def get_sponsor_link(anchor_text, bioguide_id):
    base_url = "https://bioguide.congress.gov/search/bio"
    url = "{}/{}".format(base_url, bioguide_id)
    return "[{}]({})".format(anchor_text, url)


def get_congress_gov_url(congress_num, legis_type, legis_num):
    lt = CONGRESS_GOV_TYPE_MAP[legis_type]
    return f"https://www.congress.gov/bill/{congress_num}th-congress/{lt}/{legis_num}"


def get_govtrack_url(congress_num, legis_type, legis_num):
    return (
        f"https://www.govtrack.us/congress/bills/{congress_num}/{legis_type}{legis_num}"
    )


vectorstore = load_vectorstore()


DEFAULT_PROMPT_TEMPLATE = """Use the following pieces of context from congressional legislation to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""


st.title(":classical_building: LegisQA :computer:")
st.header("Explore Congressional Legislation")
st.write(
    """When you send a question to LegisQA, it will attempt to retrieve relevant content from the past six congresses ([113th-118th covering 2013 to the present](https://en.wikipedia.org/wiki/List_of_United_States_Congresses)), pass it to a [large language model (LLM)](https://en.wikipedia.org/wiki/Large_language_model), and generate an answer. This technique is known as Retrieval Augmented Generation (RAG). You can read the [original paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) or a [recent summary](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) to get more details. Once the answer is generated, the retrieved content will be available for inspection with links to the bills and sponsors.
This technique helps to ground the LLM response by providing context from a trusted source, but it does not guarantee a high quality answer. We encourage you to play around. Try different models. Find questions that work and find questions that fail."""
)


with st.sidebar:

    st.subheader(":brain: Learn about [hyperdemocracy](https://hyperdemocracy.us)")
    st.subheader(
        ":world_map: Visualize with [nomic atlas](https://atlas.nomic.ai/data/gabrielhyperdemocracy/us-congressional-legislation-s1024o256nomic/map)"
    )
    st.subheader(":hugging_face: Explore the datasets in [huggingface](https://huggingface.co/hyperdemocracy)")

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
            "Number of chunks to retrieve",
            min_value=1,
            max_value=40,
            value=10,
        )

    with st.expander("Generative parameters"):

        temperature = st.slider("temperature", min_value=0.0, max_value=2.0, value=0.0)
        top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0)

        if llm_provider == "openai-chat":
            llm = ChatOpenAI(
                model_name=llm_name,
                temperature=temperature,
                openai_api_key=openai_api_key,
            )

        elif llm_provider == "hfhub":
            max_length = st.slider(
                "max_tokens", min_value=512, max_value=16384, value=512, step=64
            )
            llm = HuggingFaceHub(
                repo_id=llm_name,
                huggingfacehub_api_token=hfhub_api_token,
                model_kwargs={
                    "temperature": temperature,
                    "max_length": max_length,
                },
            )

        elif llm_provider == "llamacpp":
            n_ctx = st.slider(
                "n_ctx", min_value=512, max_value=16384, value=4096, step=64
            )
            max_tokens = st.slider(
                "max_tokens", min_value=512, max_value=16384, value=512, step=64
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
qa_chain = load_qa_chain(
    llm,
    chain_type="stuff",
    prompt=qa_chain_prompt,
)


col1, col2 = st.columns(2)

with col1:

    with st.form("my_form"):
        query = st.text_area("Enter question:")
        with st.expander("Filters"):
            filter_legis_id = st.text_input("Bill ID (e.g. 118-s-2293)")
            filter_bioguide_id = st.text_input("Bioguide ID (e.g. R000595)")
            filter_congress_num = st.text_input("Congress (e.g. 118)")
        submitted = st.form_submit_button("Submit")


def escape_markdown(text):
    MD_SPECIAL_CHARS = r"\`*_{}[]()#+-.!$"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\" + char)
    return text


if submitted:

    vs_filter = {}
    if filter_legis_id != "":
        vs_filter["legis_id"] = filter_legis_id
    if filter_bioguide_id != "":
        vs_filter["sponsor_bioguide_id"] = filter_bioguide_id
    if filter_congress_num != "":
        vs_filter["congress_num"] = int(filter_congress_num)

    rdocs_and_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=n_ret_docs,
        filter=vs_filter,
    )
    rdocs = [el[0] for el in rdocs_and_scores]
    rscores = [el[1] for el in rdocs_and_scores]

    if len(rdocs_and_scores) == 0:
        st.warning("No documents were retrieved. Please check the filters.")
        st.stop()

    out = qa_chain.invoke(
        {
            "input_documents": rdocs,
            "question": query,
        },
        return_only_outputs=False,
    )
    st.session_state["out"] = out


if not "out" in st.session_state:
    st.stop()
else:
    out = st.session_state["out"]


with col1:
    do_escape_markdown = st.checkbox("escape markdown in answer")
    if do_escape_markdown:
        st.info(escape_markdown(out["output_text"]))
    else:
        st.info(out["output_text"])

with col2:

    # group source documents
    grpd_source_docs = defaultdict(list)
    for doc in out["input_documents"]:
        grpd_source_docs[doc.metadata["legis_id"]].append(doc)

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

    for legis_id, doc_grp in grpd_source_docs:
        first_doc = doc_grp[0]
        ref = "{} chunks from {}\n\n{}\n\n[congress.gov]({}) | [govtrack.us]({})\n\n{}".format(
            len(doc_grp),
            first_doc.metadata["legis_id"],
            first_doc.metadata["title"],
            get_congress_gov_url(
                first_doc.metadata["congress_num"],
                first_doc.metadata["legis_type"],
                first_doc.metadata["legis_num"],
            ),
            get_govtrack_url(
                first_doc.metadata["congress_num"],
                first_doc.metadata["legis_type"],
                first_doc.metadata["legis_num"],
            ),
            get_sponsor_link(
                first_doc.metadata["sponsor_full_name"],
                first_doc.metadata["sponsor_bioguide_id"],
            ),
        )
        doc_contents = [
            "[start_index={}] ".format(doc.metadata["start_index"]) + doc.page_content
            for doc in doc_grp
        ]
        with st.expander(ref):
            st.write(escape_markdown("\n\n...\n\n".join(doc_contents)))


with st.sidebar:

    download_pack = {
        "llm_provider": llm_provider,
        "llm_name": llm_name,
        "query": query,
        "prompt_template": prompt_template,
        "out_result": out["output_text"],
        "out_source_documents": [doc.dict() for doc in out["input_documents"]],
        "time": datetime.datetime.now().isoformat(),
    }
    st.download_button(
        label="Download Results",
        data=json.dumps(download_pack, indent=4),
        file_name="legisqa_output.json",
        mime="text/json",
    )
