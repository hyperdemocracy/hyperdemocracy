import lancedb
from datasets import load_dataset
import pandas as pd
import numpy as np

from hyperdemocracy.embedding.models import BGESmallEn

class Lance:
    def __init__(self):
        self.model = BGESmallEn()
        uri = "data/sample-lancedb"
        self.db = lancedb.connect(uri)

    def create_table(self):
        ds = load_dataset("hyperdemocracy/uscb.s1024.o256.bge-small-en", split="train")
        df = pd.DataFrame(ds)
        df.rename(columns={"vec": "vector"}, inplace=True)
        table = self.db.create_table("congress", 
                                data=df)
        return 
    
    def query_table(self, queries, n=5) -> pd.DataFrame:
        q_embeddings = self.model.model.encode_queries(queries)
        table = self.db.open_table("congress")
        result = table.search(q_embeddings.reshape(384,)).limit(n).to_df()
        return result