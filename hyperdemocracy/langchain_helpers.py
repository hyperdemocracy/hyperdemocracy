"""
"""
import json
from pathlib import Path
from typing import Iterable, Union

from langchain.schema import Document


def write_docs_to_jsonl(docs: Iterable[Document], file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)
    with file_path.open("w") as fp:
        for doc in docs:
            fp.write("{}\n".format(doc.json()))


def read_docs_from_jsonl(file_path: Union[str, Path]) -> list[Document]:
    file_path = Path(file_path)
    with file_path.open("r") as fp:
        docs = []
        for line in fp:
            blob = json.loads(line.strip())
            doc = Document(**blob)
            docs.append(doc)
    return docs

