"""
https://github.com/pgvector/pgvector/tree/master
https://www.postgresql.org/docs/current/functions-json.html
https://www.psycopg.org/docs/usage.html
"""

import json
import psycopg2
from datasets import load_dataset
from tqdm import tqdm

def populate_pg(conn_dict, ds, table_name):

    num_dim = len(ds[0]["vec"])
    drop_table_sql = f"""DROP TABLE {table_name}"""
    create_table_sql = f"""
    CREATE TABLE {table_name} (
      id text PRIMARY KEY,
      text text,
      metadata JSONB,
      vec vector({num_dim})
    )
    """
    insert_row_template = f"""
    INSERT INTO {table_name} (id, text, metadata, vec) VALUES (%s, %s, %s, %s)
    """

    with psycopg2.connect(**conn_dict) as conn:
        with conn.cursor() as cursor:

            cursor.execute(drop_table_sql)
            cursor.execute(create_table_sql)
            batch_size = 64
            total = len(ds) // batch_size
            for batch in tqdm(
                ds.iter(batch_size=batch_size),
                total=total,
            ):
                batch["metadata"] = [json.dumps(el) for el in batch["metadata"]]
                values = list(zip(batch["id"], batch["text"], batch["metadata"], batch["vec"]))
                cursor.executemany(
                    insert_row_template,
                    values,
                )
                break

    conn.close()


ds_name = "hyperdemocracy/uscb.s1024.o256.bge-small-en"
embed_name = ds_name.split(".")[-1]
ds = load_dataset(ds_name, split="train")
conn_dict = {
    "database": "galtay",
    "user": "galtay",
}
table_name = "bills"
populate_pg(conn_dict, ds, table_name)



query_by_vec_template = """
SELECT id FROM bills ORDER BY vec <=> '{}' limit 1
"""


with psycopg2.connect(**conn_dict) as conn:
    with conn.cursor() as cursor:
        cursor.execute(query_by_vec_template.format(ds[2]["vec"]))
        res = cursor.fetchone()

conn.close()



