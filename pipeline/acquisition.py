from prefect import flow, task
import pandas as pd
from prefect_shell import ShellOperation
import awswrangler as wr

from hyperdemocracy.datasets.uscongress import USCongressDataset

@task 
def dataset_files(location) -> pd.DataFrame:
  pass

@task
def get_scraped_files(location):
  """
  given a location containing a USCongressDataset, will
  initialize a hyperdemocracy dataset class using that location.
  """
  # return congress_dataset

  bill_files = wr.s3.list_objects(location)
  return bill_files

@task
def get_bills_df(bill_files):
  """
  fetches the raw congress data and builds a dataset
  """
  congress_dataset = USCongressDataset(
    location='s3://hyperdemocracy-dev/congress-scrapper/data'
  ).from_files(bill_files)

  return congress_dataset.to_pandas()


@task
def write_df_to_parquet(df, location):
  """
  writes a dataframe to a parquet file
  """
  print(f"writing to {location}...")  
  wr.s3.to_parquet(
    df=df,
    path=location + '/congress.parquet',
  )



@flow(name="congress", log_prints=True)
def congress_data(): 
  # congress = USCongressDataset(base_path='s3://hyperdemocracy-dev').get_dataframe()
  raw_congress_data = get_scraped_files('s3://hyperdemocracy-dev/congress-scrapper/data/118')
  congress_dataset = get_bills_df(raw_congress_data)
  write_df_to_parquet(congress_dataset, 's3://hyperdemocracy-dev/congress')

  print(congress_dataset)

if __name__ == "__main__":
  congress_data.serve(name="congress_data")