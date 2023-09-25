import pyarrow as pa

class HyperdemocracyDataset:

  def __init__(self, location) -> None:
    self.location = location
    self.pqname = f"{self.name}.parquet"
    self.table = self._load_table()

  def _load_table(self):
    try:
      self.table = pa.parquet.read_table(self.location + '/' + self.pqname)
    except Exception as e:
      print(f"table {self.pqname} not found, initializing")
      self._initialize_table()
    
  def _initialize_table(self):
    bill_name = pa.array([], type=pa.string())
    congress_num = pa.array([], type=pa.int32())
    storage_location = pa.array([], type=pa.string())
    _hdstate = pa.array([], type=pa.string())

    self.table = pa.Table.from_arrays([bill_name, storage_location, _hdstate], names=['bill_name', 'storage_location', '_hdstate'])
    pa.parquet.write_table(self.table, self.location + '/' + self.pqname)