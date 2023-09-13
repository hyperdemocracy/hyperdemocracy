import awswrangler as wr
import pandas as pd

class LocalBackend:
    def __init__(self, path):
        self.path = path

    def save(self, df, name):
        df.to_parquet(
            path=f"{self.path}/{name}.parquet",
            index=False,
            compression="snappy",
        )

    def load(self, name):
        return pd.read_parquet(
            path=f"{self.path}/{name}.parquet",
        )

class S3Backend:
    def __init__(self, bucket, prefix):
        self.bucket = bucket
        self.prefix = prefix

    def save(self, df, name):
        wr.s3.to_parquet(
            df=df,
            path=f"s3://{self.bucket}/{self.prefix}/{name}.parquet",
            dataset=True,
            mode="overwrite",
            index=False,
            compression="snappy",
            use_threads=True,
        )

    def load(self, name):
        return wr.s3.read_parquet(
            path=f"s3://{self.bucket}/{self.prefix}/{name}.parquet",
            dataset=True,
            use_threads=True,
        )


backends = {
    's3': S3Backend,
    'local': LocalBackend,
}