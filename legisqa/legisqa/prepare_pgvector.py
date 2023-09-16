"""
https://github.com/pgvector/pgvector/tree/master
https://www.postgresql.org/docs/current/functions-json.html
https://www.psycopg.org/docs/usage.html


L2: "<->"
inner product: "<#>"
cosine distance: "<=>"

"""

import json
import psycopg2
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm


def populate_pg(
    conn_dict: dict[str, str],
    ds: Dataset,
    table_name: str,
    use_hnsw: bool,
):

    hnsw_m = 16
    hnsw_ef = 64
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

    index_vec_sql = f"""
    CREATE INDEX ON {table_name}
    USING hnsw (vec vector_cosine_ops)
    WITH (m = {hnsw_m}, ef_construction = {hnsw_ef})
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

            if use_hnsw:
                cursor.execute(index_vec_sql)

    conn.close()



def example_query(
    conn_dict: dict[str, str],
    ds: Dataset,
    ii: int,
):

    # return similarity (1-distance)
    # sort by distance in ascending order
    query_by_vec_template = """
    SELECT
      id,
      text,
      metadata,
      1 - (vec <=> '{iv}') AS score
    FROM bills
    ORDER BY vec <=> '{iv}'
    LIMIT 10
    """

    with psycopg2.connect(**conn_dict) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                query_by_vec_template.format(
                    **{"iv": ds[ii]["vec"]}
                )
            )
            res = cursor.fetchall()

    conn.close()

    return res



if __name__ == "__main__":

    ds_name = "hyperdemocracy/uscb.s1024.o256.bge-small-en"
    embed_name = ds_name.split(".")[-1]
    ds = load_dataset(ds_name, split="train")
    conn_dict = {
        "database": "galtay",
        "user": "galtay",
    }
    table_name = "bills"
    use_hnsw = True
    populate_pg(conn_dict, ds, table_name, use_hnsw)

    res = example_query(conn_dict, ds, 2)




