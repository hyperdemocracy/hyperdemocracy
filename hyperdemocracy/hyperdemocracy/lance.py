from lancedb import lancedb
from datasets import load_dataset

from hyperdemocracy.embedding.models import BGESmallEn

uri = "data/sample-lancedb"
db = lancedb.connect(uri)

# ds = load_dataset("hyperdemocracy/uscb.s1024.o256.bge-small-en", split="train")
table = db.create_table("congress", 
                        data=[{"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
                              {"vector": [5.9, 26.5], "item": "bar", "price": 20.0}])

table = db.open_table("congress")


queries = ["climate"]
# q_embeddings = model.encode_queries(queries)


result = table.search([100, 100]).limit(2).to_df()

if __name__ == "__main__":
    print(result)