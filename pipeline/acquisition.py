from prefect import flow, task
import pandas as pd
from prefect_shell import ShellOperation
import awswrangler as wr

from hyperdemocracy.datasets.uscongress import USCongressDataset

@task
def get_raw_congress_data() -> pd.DataFrame:
  """
  returns a list of file paths to congress data saved to disk
  """

  bill_files = wr.s3.list_objects('s3://hyperdemocracy-dev/congress-scrapper/data/govinfo/BILLS')
  return bill_files

@task
def build_dataset(bill_files: list):
  """
  fetches the raw congress data and builds a dataset
  """
  congress_dataset = USCongressDataset(base_path='s3://hyperdemocracy-dev', backend='s3').from_files(bill_files)
  return congress_dataset
  # for file in bill_files:
  #   print(f"processing {file}")

  #   congress_dataset = USCongressDataset(base_path='s3://hyperdemocracy-dev').from_files()



@flow(name="congress", log_prints=False)
def congress_data(): 
  # congress = USCongressDataset(base_path='s3://hyperdemocracy-dev').get_dataframe()
  raw_congress_data = get_raw_congress_data()
  congress_dataset = build_dataset(raw_congress_data)
  print(len(congress_dataset.bills))

if __name__ == "__main__":
  congress_data.serve(name="congress_data")