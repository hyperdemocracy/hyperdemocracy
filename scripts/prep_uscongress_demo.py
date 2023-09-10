"""

"""

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import pandas as pd

from hyperdemocracy import langchain_helpers


chunk_size = 1024
chunk_overlap = 256


base_hd_path = Path("/home/galtay/data/hyperdemocracy")
docs_path = base_hd_path / "lc_docs" / "lc_docs.jsonl"
docs = langchain_helpers.read_docs_from_jsonl(docs_path)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap,
    add_start_index = True,
)
split_docs = text_splitter.split_documents(docs)


model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedder = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vecs = embedder.embed_documents([doc.page_content for doc in split_docs])
rows = []
for doc, vec in zip(split_docs, vecs):
    row = {
        "text": doc.page_content,
        "metadata": doc.metadata,
        "vec": vec,
    }
    rows.append(row)

df = pd.DataFrame(rows)
