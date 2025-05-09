from abc import ABC, abstractmethod
from typing import Literal

class BaseDataset(ABC):

  """
    Base abstract class for the Datasets
  """
  @abstractmethod
  def dataset_load(self):
    pass


class CSVDataset(BaseDataset):
  """
    CSVDataset handles the logic to load CSV Datasets
  """
  def __init__(self, path, **kwargs):
    """
      Args:
        - path: path to csv file
        - optional args for pandas read_csv method
    """
    self.path = path
    self.kwargs = kwargs

  def dataset_load(self):
    """
      Method that loads and returns the CSV dataset
    """
    import pandas as pd
    df = pd.read_csv(self.path, **self.kwargs)
    return df


class HFDataset(BaseDataset):
  """
    HFDataset handles the logic to load HuggingFace Datasets
  """

  def __init__(self, path, **kwargs):
    """
      Args:
        - path: path to Huggingface dataset
        - optional args for datasets.load_dataset() method
    """

    self.path = path
    self.kwargs = kwargs

  def dataset_load(self):
    """
      Method that loads and returns the HuggingFace dataset
    """

    from datasets import load_dataset
    dataset = load_dataset(self.path, **self.kwargs)
    return dataset


class DatasetCreator():
  """
    DatasetCreator is an abstraction that generates the correct Dataset type.
  """
  @staticmethod
  def create_dataset(dataset_type: Literal["csv", "hf"], path: str, **kwargs):
    """
    Args:
      - dataset_type: can be "csv" or "hf"
      - path: path to csv file or HuggingFace dataset
      - optional arguments: passed to pandas.read_csv() / datasets.load_dataset()
    """
    if(dataset_type == "csv"):
      return CSVDataset(path, **kwargs)
    elif(dataset_type == "hf"):
      return HFDataset(path, **kwargs)
    else:
      raise Exception("Invalid Dataset type")
    
class DatasetLoader():
  def __init__(self, dataset_type, path, **kwargs):
    self.dataset = DatasetCreator().create_dataset(dataset_type, path, **kwargs)
  def load(self):
    return self.dataset.dataset_load()