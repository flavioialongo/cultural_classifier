import torch
import re
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from datasets import Dataset as HFaceDataset 
import os 
import pandas as pd 

class WikiGraphDataset(torch.utils.data.Dataset):
  """
    Torch Dataset that concatenates the encoded Wikipedia text and Wikidata graph
    relative to each item in the given dataset.
  """
  def __init__(self, dataset: Union[HFaceDataset, pd.DataFrame], data_folder: str,  is_test: bool, graph = None, wikidict = None):
    """
      Initializes the WikiGraphDataset for the given dataset
    """

    self.pattern = re.compile(r"\W+")
    self.LABEL2ID =  {
      "cultural representative": 0,
      "cultural agnostic": 1,
      "cultural exclusive": 2,
      "no_label": -1
    }

    self.is_test = is_test
    self.dataset = dataset
    self.data_folder = data_folder

    self.is_pandas = (not isinstance(dataset, HFaceDataset))

    # Load the data structures automatically from files in the folder
    if(graph == None):
      self.graph_dict = self._load_data("graph_dictionary.pkl")
    else:
      self.graph_dict = graph

    if(wikidict == None):
      self.wikidict = self._load_data("wikidict.pkl")
    else:
      self.wikidict = wikidict

    self.w2v_key2index = self._load_data("w2v_key2index.pkl")
    self.graph2idx = self._load_data("graph2idx.pkl")

    self.graph_embedding = torch.load(self.data_folder+"graph_embedding.pt")
    self.w2v_embedding = torch.load(self.data_folder+"word2vec_embedding.pt")

    # Build Wikipedia sample structure
    self.wiki_samples = self._get_wiki_samples()

    # Build Graph map
    self.graph_encoding = self._get_graph_encoding()


    #Final sample structure

    #If test set, ignore target tensor
    if(self.is_test):
      self.samples = [
        {
            "input": sample["input"],
            "graph": self.graph_encoding[sample["qid"]]
        }
        for sample in self.wiki_samples
        if sample["qid"] in self.graph_encoding
      ]
    else:
      self.samples = [
        {
            "input": sample["input"],
            "target": sample["target"],
            "graph": self.graph_encoding[sample["qid"]]
        }
        for sample in self.wiki_samples
        if sample["qid"] in self.graph_encoding
      ]



  def _get_wiki_samples(self):
    """
      Helper method that retrieves Wikipedia articles for each item in the dataset row.
      It uses the files
        - wikidict (dict): a dictionary with the structure QID: Wikipedia Article.
        - W2V_key2index (dict): a mapping between words and Pre-Trained Word2Vec
      Returns:
        - wiki_samples: Array of dictionaries with the following keys
              - input (torch.tensor): torch tensor that holds the entire article words encoded using Word2Vec key-to-index
              - target (torch.tensor): Encoded label for the given row.
              - qid (str): Wikidata entity ID (e.g. Pizza -> Q177)

    """
    wiki_samples = []

    iterator = self.dataset

    #Pandas Dataframe
    if(self.is_pandas):
      iterator = self.dataset.iterrows()

    for row in iterator:

      if(not self.is_pandas):
        qid = row["item"].split("/")[-1]
      else:
        row = row[1]
        qid = row["item"].split("/")[-1]

      if(self.is_test):
        label_str = "no_label"
      else:
        label_str = row["label"]

      if qid not in self.wikidict or label_str not in self.LABEL2ID:
        continue

      text = self.wikidict[qid]
      tokens = [w for w in self.pattern.split(text.lower()) if w]
      encoded = [self.w2v_key2index[w] for w in tokens if w in self.w2v_key2index]

      if not encoded:
        continue

      # if test set, we dont'have target
      if(self.is_test):
        wiki_samples.append({
            "input": torch.tensor(encoded, dtype=torch.long),
            "qid": qid
        })
      else:
        wiki_samples.append({
            "input": torch.tensor(encoded, dtype=torch.long),
            "target": torch.tensor(self.LABEL2ID[label_str], dtype=torch.long),
            "qid": qid
        })

    return wiki_samples


  def _get_graph_encoding(self):

    """
      Helper method that retrieves Graph Encoding.
      It uses the files
        - graph_dict (dict): A dictionary where each key is a Wikidata entity ID (QID),
                          and the value is another dictionary containing:
                          - "properties" (list): A list of properties associated with the entity (e.g. both prop in item->prop->entity1->prop->entity2).
                          - "entities" (list): A list of other entities connected to the entity (e.g. both entities in item->prop->entity1->prop->entity2).
        - graph2idx (dict): a mapping between nodes in the graph and Pre-Trained Node2Vec

      Returns:
        - graph_encoding (dict): A dictionary with key QID and values the array of encoded nodes for that QID.
    """

    graph_encoding = {}
    for qid in [sample["qid"] for sample in self.wiki_samples]:
      if qid in self.graph_dict:
        indices = []
        if qid in self.graph2idx:
          indices.append(self.graph2idx[qid])

        for prop in self.graph_dict[qid].get("properties"):
          if prop in self.graph2idx:
            indices.append(self.graph2idx[prop])

        for entity in self.graph_dict[qid].get("entities"):
          if entity in self.graph2idx:
            indices.append(self.graph2idx[entity])

        if indices:
          graph_encoding[qid] = torch.tensor(indices, dtype=torch.long)

    return graph_encoding

  def _load_data(self, filename):
    import pickle

    """
    Helper method to load data from files.

    Args:
        filename (str): The name of the file to load.

    Returns:
        The loaded data object (e.g., dictionary, list, etc.).
    """
    file_path = os.path.join(self.data_folder, filename)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"File {filename} not found in {self.data_folder}")

  def __getitem__(self, idx):
    return self.samples[idx]

  def __len__(self):
    return len(self.samples)


  def collate_fn(self, batch):
    inputs = [b["input"] for b in batch]
    graphs = [b["graph"] for b in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_graphs = pad_sequence(graphs, batch_first=True, padding_value=0)

    if(not self.is_test):
      targets = [b["target"] for b in batch]
      targets = torch.tensor(targets, dtype=torch.long)
      return {"wiki": padded_inputs, "graph": padded_graphs, "target": targets}
    else:
      return {"wiki": padded_inputs, "graph": padded_graphs}
