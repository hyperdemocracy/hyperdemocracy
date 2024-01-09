# Import necessary libraries and modules
from collections import defaultdict  # For creating dictionaries with default values
import datetime  # For working with dates and times
import json  # For working with JSON data
import os  # For interacting with the operating system
from pathlib import Path  # For creating and manipulating filesystem paths

# Import specific modules from the langchain package
import chromadb  # Database module
import streamlit as st  # Streamlit for creating web apps
from langchain.embeddings import HuggingFaceBgeEmbeddings  # For creating embeddings using HuggingFace
from langchain.vectorstores import Chroma  # For storing and retrieving vectors
from langchain.chat_models import ChatOpenAI  # For creating chat models using OpenAI
from langchain import HuggingFaceHub  # For interacting with the HuggingFace Hub
from langchain.llms import LlamaCpp  # For working with LlamaCpp
from langchain.chains import RetrievalQA  # For retrieving question and answer chains
from langchain.prompts import PromptTemplate  # For creating prompt templates
from langchain.chains.question_answering import load_qa_chain  # For loading question and answer chains
import openai  # For interacting with OpenAI

# Set up the Streamlit page configuration
st.set_page_config(layout="wide", page_title="LegisQA")  # Set the layout to "wide" and the page title to "LegisQA"

# Get environment variables
env_openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the OpenAI API key from the environment variables
env_hfhub_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Get the HuggingFace Hub API token from the environment variables
env_gguf_path = os.getenv("GGUF_PATH")  # Get the GGUF path from the environment variables

# Define a list of LLM providers
LLM_PROVIDERS = ["openai-chat", "hfhub", "llamacpp"]  # The available LLM providers are "openai-chat", "hfhub", and "llamacpp"

# Define a list of OpenAI chat models
OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
]  # The available OpenAI chat models are "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", and "gpt-4-32k"

# Function to load the vector store
def load_vectorstore():
    # Define the model name and arguments
    model_name = "BAAI/bge-small-en"  # The name of the model
    model_kwargs = {'device': 'cpu'}  # The arguments for the model
    encode_kwargs = {'normalize_embeddings': True}  # The arguments for encoding

    # Create an embedder using HuggingFace
    embedder = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )  # The embedder is created with the specified model name and arguments

    # Create a Chroma client
    chroma_client = chromadb.PersistentClient(path="chroma.db")  # The Chroma client is created with the specified path

    # Get the collection from the Chroma client
    collection = chroma_client.get_collection(
        name='uscb',
        embedding_function=embedder.embed_documents,
    )  # The collection is retrieved from the Chroma client using the specified name and embedding function

    # Create a Chroma vector store
    vectorstore = Chroma(
        collection_name="uscb",
        embedding_function=embedder,
        persist_directory="chroma.db",
        client=chroma_client,
    )  # The vector store is created with the specified collection name, embedding function, persist directory, and client
    # Return the vector store
    return vectorstore  # The vector store is returned for further use

# Function to generate a link to the sponsor's bio
def get_sponsor_link(sponsors):
    base_url = "https://bioguide.congress.gov/search/bio"  # Base URL for the bio search
    dd = json.loads(sponsors)[0]  # Load the first sponsor from the JSON data
    url = "{}/{}".format(base_url, dd["bioguideId"])  # Format the URL with the sponsor's bioguide ID
    return "[{}]({})".format(dd["fullName"], url)  # Return a markdown link with the sponsor's full name and URL

# Load the vector store
vectorstore = load_vectorstore()  # The vector store is loaded for use in the application

# Define the default prompt template
DEFAULT_PROMPT_TEMPLATE = """Use the following pieces of context from congressional legislation to answer the question at the end.
Remember that you can only answer questions about the content of the legislation. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""  # This is the default template for prompts sent to the language model

# Set the title of the Streamlit app
st.title(":classical_building: LegisQA :computer:")  # The title is set to "LegisQA"

# Set the header of the Streamlit app
st.header("Explore the Legislation of the 118th US Congress")  # The header is set to "Explore the Legislation of the 118th US Congress"

# Write a description of the Streamlit app
st.write(
    """When you send a question to LegisQA, it will attempt to retrieve relevant content from the [118th United States Congress](https://en.wikipedia.org/wiki/118th_United_States_Congress), pass it to a [large language model (LLM)](https://en.wikipedia.org/wiki/Large_language_model), and generate an answer. This technique is known as Retrieval Augmented Generation (RAG). You can read the [original paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) or a [recent summary](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) to get more details. Once the answer is generated, the retrieved content will be available for inspection with links to the bills and sponsors.
This technique helps to ground the LLM response by providing context from a trusted source, but it does not guarantee a high quality answer. We encourage you to play around. Try different models. Find questions that work and find questions that fail."""  # This description explains how the app works and encourages users to experiment with different models and questions
)

# Create a sidebar in the Streamlit app
with st.sidebar:
    # Add a subheader to the sidebar
    st.subheader(":brain: Learn about [hyperdemocracy](https://hyperdemocracy.us)")  # The subheader contains a link to learn more about hyperdemocracy

    # Add another subheader to the sidebar
    st.subheader(":world_map: Visualize with [nomic atlas](https://atlas.nomic.ai/map/b65c568b-ce37-40a1-b376-b20cc7580118/91b3337c-f6c9-4c1c-a755-18c84aa4141c)")  # The subheader contains a link to visualize with nomic atlas

    # Add a divider to the sidebar
    st.divider()  # This creates a visual separation in the sidebar

    # Add a select box to the sidebar to choose the LLM provider
    llm_provider = st.selectbox(
        label="llm provider",
        options=LLM_PROVIDERS,
    )  # The user can select an LLM provider from the options defined earlier

    # If the selected LLM provider is "openai-chat"
    if llm_provider == "openai-chat":
        # Define the options for the LLM name
        llm_name_options = OPENAI_CHAT_MODELS  # The options are the OpenAI chat models defined earlier

        # Add a select box to the sidebar to choose the LLM name
        llm_name = st.selectbox(
            label="llm",
            options=llm_name_options,
        )  # The user can select an LLM name from the options defined earlier

        # If the OpenAI API key is not set in the environment variables
        if env_openai_api_key is None:
            # Add a text input to the sidebar to enter the OpenAI API key
            openai_api_key = st.text_input(
                "Provide your OpenAI API key here (sk-...)",
                type="password",
            )  # The user can enter their OpenAI API key, which is hidden as it is a password
        else:
            # Use the OpenAI API key from the environment variables
            openai_api_key = env_openai_api_key

        # If the OpenAI API key is not provided
        if openai_api_key == "":
            # Stop the execution of the Streamlit app
            st.stop()  # This prevents the app from running without the necessary API key

        # If the LLM provider is "hfhub"
    elif llm_provider == "hfhub":
        # Add a text input to the sidebar to enter the HF model name
        llm_name = st.text_input(
            "Provide a HF model name (google/flan-t5-large)",
        )  # The user can enter the HF model name

        # If the HFHub API key is not set in the environment variables
        if env_hfhub_api_key is None:
            # Add a text input to the sidebar to enter the HFHub API token
            hfhub_api_token = st.text_input(
                "Provide your HF API token here (hf_...)",
                type="password",
            )  # The user can enter their HFHub API token, which is hidden as it is a password
        else:
           # Use the HFHub API key from the environment variables
            hfhub_api_token = env_hfhub_api_key

        # If the HFHub API token or the HF model name is not provided
        if hfhub_api_token == "" or llm_name == "":
            # Stop the execution of the Streamlit app
            st.stop()  # This prevents the app from running without the necessary API token or model name
                
    # If the LLM provider is "llamacpp"
    elif llm_provider == "llamacpp":
        # If the GGUF path is not set in the environment variables
        if env_gguf_path is None:
            # Add a text input to the sidebar to enter the GGUF path
            gguf_path = st.text_input("Provide a path to *.gguf files")  # The user can enter the GGUF path
        else:
            # Use the GGUF path from the environment variables
            gguf_path = env_gguf_path

        # If the GGUF path is not provided
        if gguf_path == "":
            # Stop the execution of the Streamlit app
            st.stop()  # This prevents the app from running without the necessary GGUF path
        else:
            # Convert the GGUF path to a Path object
            gguf_path = Path(gguf_path)

        # If the GGUF path does not exist
        if not gguf_path.exists():
            # Display a warning message
            st.warning("Provided gguf path does not exists")  # This warns the user that the provided GGUF path does not exist
            # Stop the execution of the Streamlit app
            st.stop()  # This prevents the app from running with a non-existent GGUF path

        # Get a list of all GGUF files in the GGUF path
        gguf_files = sorted(list(gguf_path.glob("*.gguf")))

        # If there are no GGUF files in the GGUF path
        if len(gguf_files) == 0:
            # Display a warning message
            st.warning("Provided gguf path contains no gguf files")  # This warns the user that the provided GGUF path contains no GGUF files
            # Stop the execution of the Streamlit app
            st.stop()  # This prevents the app from running without any GGUF files

        # Create a map of GGUF file names to GGUF files
        gguf_map = {gf.name: gf for gf in gguf_files}

        # Get the keys of the GGUF map as the options for the LLM name
        llm_name_options = gguf_map.keys()

        # Add a select box to the sidebar to choose the LLM name
        llm_name = st.selectbox(
            label="llm",
            options=llm_name_options,
        )  # The user can select an LLM name from the options defined earlier

    # Create an expander for retrieval parameters
    with st.expander("Retrieval parameters"):
        # Add a slider to the expander to choose the number of chunks to retrieve
        n_ret_docs = st.slider(
            'Number of chunks to retrieve',
            min_value=1,
            max_value=40,
            value=10,
        )  # The user can select the number of chunks to retrieve using the slider

    # Create an expander for generative parameters
    with st.expander("Generative parameters"):
        # Add a slider to the expander to choose the temperature
        temperature = st.slider('temperature', min_value=0.0, max_value=2.0, value=0.0)  # The user can select the temperature using the slider

        # Add a slider to the expander to choose the top_p value
        top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=1.0)  # The user can select the top_p value using the slider

        # If the LLM provider is "openai-chat"
        if llm_provider == "openai-chat":
            # Create an instance of the ChatOpenAI
            llm = ChatOpenAI(
                model_name=llm_name,  # The model name is the LLM name
                temperature=temperature,  # The temperature for the model
                openai_api_key=openai_api_key,  # The OpenAI API key
            )  # This uses the provided parameters to create the ChatOpenAI instance

        # If the LLM provider is "hfhub"
        elif llm_provider == "hfhub":
            # Add a slider to the expander to choose the maximum number of tokens
            max_length = st.slider(
                'max_tokens',
                min_value=512,
                max_value=16384,                    value=512,
                step=64
            )  # The user can select the maximum number of tokens using the slider

            # Create an instance of the HuggingFaceHub
            llm = HuggingFaceHub(
                repo_id=llm_name,  # The repository ID is the LLM name
                huggingfacehub_api_token=hfhub_api_token,  # The API token for HuggingFaceHub
                model_kwargs={
                    "temperature": temperature,  # The temperature for the model
                    "max_length": max_length,  # The maximum number of tokens for the model
                }
            )  # This uses the provided parameters to create the HuggingFaceHub instance

        # If the LLM provider is "llamacpp"
        elif llm_provider == "llamacpp":
            # Add a slider to the expander to choose the n_ctx value
            n_ctx = st.slider(
                'n_ctx',
                min_value=512,
                max_value=16384,
                value=4096,
                step=64
            )  # The user can select the n_ctx value using the slider

            # Add a slider to the expander to choose the maximum number of tokens
            max_tokens = st.slider(
                'max_tokens',
                min_value=512,
                max_value=16384,
                value=512,
                step=64
            )  # The user can select the maximum number of tokens using the slider

            # Create an instance of the LlamaCpp
            llm = LlamaCpp(
                model_path=str(gguf_map[llm_name]),  # The model path is the GGUF map for the LLM name
                temperature=temperature,  # The temperature for the model
                max_tokens=max_tokens,  # The maximum number of tokens for the model
                top_p=top_p,  # The top_p value for the model
                n_ctx=n_ctx,  # The n_ctx value for the model
            )  # This uses the provided parameters to create the LlamaCpp instance

    # Create an expander for the prompt
    with st.expander("Prompt"):
        # Add a text area to the expander to enter the prompt template
        prompt_template = st.text_area(
            "prompt template",
            DEFAULT_PROMPT_TEMPLATE,
            height=300,
        )  # The user can enter the prompt template, which defaults to the default prompt template

# Create a PromptTemplate object from a predefined template
qa_chain_prompt = PromptTemplate.from_template(prompt_template)
qa_chain = load_qa_chain(
    llm,
    chain_type="stuff",
    prompt=qa_chain_prompt,
)

# Create two columns
col1, col2 = st.columns(2)

# In the first column
with col1:
    # Create a form
    with st.form("my_form"):
        # Add a text area to the form to enter a question
        query = st.text_area('Enter question:')
                
        # Create an expander for filters
        with st.expander("Filters"):
            # Add a text input to the expander to enter the Bill ID
            filter_bill_id = st.text_input("Bill ID (e.g. 118-S-2293)")
                
        # Add a submit button to the form
        submitted = st.form_submit_button('Submit')

# Define a function to escape markdown characters in a text
def escape_markdown(text):
    # Define the special characters in markdown
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!$"
    
    # For each special character
    for char in MD_SPECIAL_CHARS:
        # Replace the character with its escaped version
        text = text.replace(char, "\\"+char)
    
    # Return the escaped text
    return text

# If the form was submitted
if submitted:
    # Initialize the filter for the vectorstore as None
    vs_filter = None
    
    # If a Bill ID was provided
    if filter_bill_id != "":
        # Set the filter for the vectorstore to the provided Bill ID
        vs_filter = {"parent_id": filter_bill_id}

    # Perform a similarity search in the vectorstore with the provided query, number of returned documents, and filter
    rdocs_and_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=n_ret_docs,
        filter=vs_filter,
    )
    
    # Extract the retrieved documents and their scores from the results
    rdocs = [el[0] for el in rdocs_and_scores]
    rscores = [el[1] for el in rdocs_and_scores]

    # If no documents were retrieved
    if len(rdocs_and_scores) == 0:
        # Display a warning message
        st.warning("No documents were retrieved. Please check the filters.")
        # Stop the execution of the Streamlit app
        st.stop()

    # Perform the QA chain with the retrieved documents and the provided question
    out = qa_chain(
        {
            "input_documents": rdocs,
            "question": query,
        },
        return_only_outputs=False,
    )
    
    # Store the output in the session state
    st.session_state["out"] = out

# If there is no output in the session state
if not "out" in st.session_state:
    # Stop the execution of the Streamlit app
    st.stop()
# Otherwise
else:
    # Retrieve the output from the session state
    out = st.session_state["out"]

# In the first column
with col1:
    # Add a checkbox to decide whether to escape markdown in the answer
    do_escape_markdown = st.checkbox("escape markdown in answer")
    
    # If the checkbox is checked
    if do_escape_markdown:
        # Display the escaped output text
        st.info(escape_markdown(out["output_text"]))
    # Otherwise
    else:
        # Display the output text as is
        st.info(out["output_text"])

# In the second column
with col2:
    # Initialize a dictionary to group source documents by their parent ID
    grpd_source_docs = defaultdict(list)
    
    # For each document in the input documents
    for doc in out["input_documents"]:
        # Add the document to its group based on its parent ID
        grpd_source_docs[doc.metadata["parent_id"]].append(doc)

    # For each group
    for key in grpd_source_docs:
        # Sort the documents in the group by their start index
        grpd_source_docs[key] = sorted(
            grpd_source_docs[key],
            key=lambda x: x.metadata["start_index"],
        )

    # Sort the groups by the number of documents in each group
    grpd_source_docs = sorted(
        tuple(grpd_source_docs.items()),
        key=lambda x: -len(x[1]),
    )

    # For each group and its documents
    for parent_id, doc_grp in grpd_source_docs:
        # Get the first document in the group
        first_doc = doc_grp[0]
        
        # Create a reference string with the number of documents in the group, the parent ID, the title, the URLs, and the sponsors
        ref = "{} chunks from {}\n\n{}\n\n[congress.gov]({}) | [govtrack.us]({})\n\n{}".format(
            len(doc_grp),
            first_doc.metadata["parent_id"],
            first_doc.metadata["title"],
            first_doc.metadata["congress_gov_url"],
            first_doc.metadata["govtrack_url"],
            get_sponsor_link(first_doc.metadata["sponsors"]),
        )
        
        # Create a list of the contents of the documents in the group
        doc_contents = [
            "[start_index={}] ".format(doc.metadata["start_index"]) + doc.page_content
            for doc in doc_grp
        ]
        
        # Create an expander with the reference string
        with st.expander(ref):
            # Display the escaped contents of the documents in the expander
            st.write(
                escape_markdown("\n\n...\n\n".join(doc_contents))
            )

# Within the sidebar of the Streamlit app
with st.sidebar:
    # Create a dictionary to hold the data to be downloaded
    download_pack = {
        "llm_provider": llm_provider,  # The LLM provider
        "llm_name": llm_name,  # The LLM name
        "query": query,  # The query
        "prompt_template": prompt_template,  # The prompt template
        "out_result": out["output_text"],  # The output text
        "out_source_documents": [doc.dict() for doc in out["input_documents"]],  # The input documents
        "time": datetime.datetime.now().isoformat(),  # The current time
    }
    
    # Add a download button to the sidebar
    st.download_button(
        label="Download Results",  # The label of the button
        data = json.dumps(download_pack, indent=4),  # The data to be downloaded, which is the download pack converted to a JSON string
        file_name='legisqa_output.json',  # The name of the downloaded file
        mime='text/json',  # The MIME type of the downloaded file
    )  # This creates a download button that downloads the download pack as a JSON file
