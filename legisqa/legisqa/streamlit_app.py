from collections import defaultdict
import datetime
import json
import os
from pathlib import Path

import chromadb
import streamlit as st
import openai
from pinecone import Pinecone as PineconeClient

from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain



st.set_page_config(layout="wide", page_title="LegisQA")


env_openai_api_key = os.getenv("LEGISQA_OPENAI_API_KEY")
env_hfhub_api_key = os.getenv("LEGISQA_HUGGINGFACEHUB_API_TOKEN")
env_gguf_path = os.getenv("LEGISQA_GGUF_PATH")
env_chroma_path = os.getenv("LEGISQA_CHROMA_PATH")
env_pinecone_api_key = os.getenv("LEGISQA_PINECONE_API_KEY")


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


LLM_PROVIDERS = ["openai", "hfhub", "llamacpp"]
OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
]


def load_chroma_vectorstore():
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    emb_fn = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this question for searching relevant passages: ",
    )

    collection_name = "usc-113-to-118-vecs-v1-s1024-o256-BAAI-bge-small-en-v1.5"
    chroma_client = chromadb.PersistentClient(path=env_chroma_path)
    collection = chroma_client.get_collection(name=collection_name)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=emb_fn,
    )
    return vectorstore


def load_pinecone_vectorstore():
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    emb_fn = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this question for searching relevant passages: ",
    )

    index_name = "usc-113to118-s1024-o256-bge-small-en-v1p5"
    pinecone = PineconeClient(
        api_key=env_pinecone_api_key,
#        environment=PINECONE_ENVIRONMENT,
    )
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=emb_fn,
    )

    return vectorstore


def get_sponsor_url(bioguide_id):
    return f"https://bioguide.congress.gov/search/bio/{bioguide_id}"


def get_congress_gov_url(congress_num, legis_type, legis_num):
    lt = CONGRESS_GOV_TYPE_MAP[legis_type]
    return f"https://www.congress.gov/bill/{congress_num}th-congress/{lt}/{legis_num}"


def get_govtrack_url(congress_num, legis_type, legis_num):
    return (
        f"https://www.govtrack.us/congress/bills/{congress_num}/{legis_type}{legis_num}"
    )


#vectorstore = load_chroma_vectorstore()
vectorstore = load_pinecone_vectorstore()


DEFAULT_PROMPT_TEMPLATE = """Use the following snippets from congressional legislation to answer the question at the end. Each snippet header contains a snippet_num index, a unique legis_id, the title of the legislation, and a short text snippet. When answering the question, refer to the legis_id and title in snippets that are useful for answering the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

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
    st.subheader(":hugging_face: Explore the [huggingface datasets](https://huggingface.co/hyperdemocracy)")

    st.divider()

    with st.container(border=True):

        llm_provider = st.selectbox(
            label="llm provider",
            options=LLM_PROVIDERS,
        )

        if llm_provider == "openai":
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

        if llm_provider == "openai":
            llm = ChatOpenAI(
                model_name=llm_name,
                temperature=temperature,
                openai_api_key=openai_api_key,
                model_kwargs={"top_p": top_p},
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


def format_docs_v1(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs_v2(docs):

    def one_doc(idoc, doc):
        return f"snippet_num={idoc}\nlegis_id={doc.metadata['legis_id']}\ntitle={doc.metadata['title']}\n... {doc.page_content} ...\n"

    snips = []
    for idoc, doc in enumerate(docs):
        txt = one_doc(idoc, doc)
        snips.append(txt)

    return "\n===\n".join(snips)


format_docs = format_docs_v2


if submitted:

    vs_filter = {}
    if filter_legis_id != "":
        vs_filter["legis_id"] = filter_legis_id
    if filter_bioguide_id != "":
        vs_filter["sponsor_bioguide_id"] = filter_bioguide_id
    if filter_congress_num != "":
        vs_filter["congress_num"] = int(filter_congress_num)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": n_ret_docs, "filter": vs_filter},
    )

    prompt = PromptTemplate.from_template(prompt_template)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    out = rag_chain_with_source.invoke(query)


    st.session_state["out"] = out


if not "out" in st.session_state:
    st.stop()
else:
    out = st.session_state["out"]


with col1:
    do_escape_markdown = st.checkbox("escape markdown in answer")
    if do_escape_markdown:
        st.info(escape_markdown(out["answer"]))
    else:
        st.info(out["answer"])

with col2:

    # group source documents
    grpd_source_docs = defaultdict(list)
    for doc in out["context"]:
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
        ref = "{} chunks from {}\n\n{}\n\n[congress.gov]({}) | [govtrack.us]({})\n\n[{} ({}) ]({})".format(
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
            first_doc.metadata["sponsor_full_name"],
            first_doc.metadata["sponsor_bioguide_id"],
            get_sponsor_url(first_doc.metadata["sponsor_bioguide_id"]),
        )
        doc_contents = [
            "[start_index={}] ".format(doc.metadata["start_index"]) + doc.page_content
            for doc in doc_grp
        ]
        with st.expander(ref):
            st.write(escape_markdown("\n\n...\n\n".join(doc_contents)))

st.text(format_docs(out["context"]))


with st.sidebar:

    download_pack = {
        "llm_provider": llm_provider,
        "llm_name": llm_name,
        "query": query,
        "prompt_template": prompt_template,
        "out_result": out["answer"],
        "out_source_documents": [doc.dict() for doc in out["context"]],
        "time": datetime.datetime.now().isoformat(),
    }
    st.download_button(
        label="Download Results",
        data=json.dumps(download_pack, indent=4),
        file_name="legisqa_output.json",
        mime="text/json",
    )
