import json
from datasets import load_dataset
from nomic import atlas
from nomic import AtlasProject
import numpy as np



def get_sponsor_url(sponsors):
    base_url = "https://bioguide.congress.gov/search/bio"
    dd = sponsors[0]
    url = "{}/{}".format(base_url, dd["bioguideId"])
    return url

def get_sponsor_name(sponsors):
    dd = sponsors[0]
    return dd["fullName"]



ds_name = "hyperdemocracy/uscb.s1024.o256.bge-small-en"
project_name = ds_name.split("/")[-1]
embed_name = ds_name.split(".")[-1]
ds = load_dataset(ds_name, split="train")
df = ds.to_pandas()

num_points = len(ds)
embeddings = np.array(df.head(num_points)["vec"].to_list())
data = [
    {
        "id": row["id"],
        "title": row["metadata"]["title"],
        "bill_url": row["metadata"]["congress_gov_url"],
        "sponsor_name": get_sponsor_name(row["metadata"]["sponsors"]),
        "sponsor_url": get_sponsor_url(row["metadata"]["sponsors"]),
        "text": row["text"],
        "policy_area": row["metadata"]["policy_area"],
        "subjects": "|".join(row["metadata"]["subjects"]),
    } for _, row in df.head(num_points).iterrows()
]


project = atlas.map_embeddings(
    embeddings=embeddings,
    data=data,
    id_field="id",
    name=project_name,
    colorable_fields=["policy_area"],
    topic_label_field="text",
    description="Legislation from the 118th US Congress",
    reset_project_if_exists=False,
)
