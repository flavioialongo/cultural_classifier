from datasets import Dataset as HFaceDataset 
import pandas as pd
import re 
from tqdm import tqdm 


class GraphCreator():
  """
    GraphCreator class generates the graph-like structure for the Wikidata item
    Attributes:
      - dataset: given dataset
      - graph: graph structure
  """
  def __init__(self, dataset):
    from wikidata.client import Client

    self.dataset = dataset
    self.client = Client()


    if(isinstance(dataset, HFaceDataset)):
      self.entities = [x.split("/")[-1] for x in dataset["item"]]

    elif(isinstance(dataset, pd.DataFrame)):
      items = dataset["item"].tolist()  # For pandas DataFrame
      self.entities = [x.split("/")[-1] for x in items]
    else:
      raise Exception("Unknown dataset type")

    self.graph = self._retrieve_graph()


  def _retrieve_graph(self):

    # Build the graph by extracting level 1 and level 2 entities
    level1 = self._extract_level1(self.entities)
    level2 = self._extract_level2(level1)

    # Convert level data into a graph dictionary
    level1_dic = self._level_to_dictionary(level1)
    level2_dic = self._level_to_dictionary(level2)

    # Merge levels into one graph
    return self._merge_dictionaries(level1_dic, level2_dic)

  def _fetch_entity(self, qid):
    # Fetch the entity's claims (properties) from Wikidata
    try:
      entity = self.client.get(qid, load=True)
      claims = entity.attributes.get("claims", {})
      return qid, self._extract_properties(claims)
    except Exception as e:
      print(f"Error on {qid}: {e}")
      return qid, {}

  def _extract_properties(self, claims):
    # Extract only useful property values from claims
    result = {}
    for prop, claim_list in claims.items():
        values = [
            claim.get("mainsnak", {}).get("datavalue", {}).get("value")
            for claim in claim_list
            if claim.get("mainsnak", {}).get("datavalue", {}).get("value") is not None
        ]
        if values:
            result[prop] = values
    return result

  def _extract_level1(self, qid_list):
    # Extracts first level of Wikidata: entity->prop->entity
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Extracting Level 1...")
    data = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(self._fetch_entity, qid): qid for qid in qid_list}
        for future in tqdm(as_completed(futures), total=len(futures)):
            qid, prop_vals = future.result()
            data[qid] = prop_vals
    return data

  def _extract_level2(self, level1_dict):
    # From the first level if the property is linked to another entity
    # retrieve other level -> entity1->prop-> entity2 (level1)
    #                         entity2->prop->entity3 (level2)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidate_qids = {
        val["id"]
        for prop_dict in level1_dict.values()
        for values in prop_dict.values()
        for val in values
        if isinstance(val, dict) and "id" in val
    }

    print(f"Extracting Level 2 from {len(candidate_qids)} unique entity values...")
    data = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(self._fetch_entity, qid): qid for qid in candidate_qids}
        for future in tqdm(as_completed(futures), total=len(futures)):
            qid, prop_vals = future.result()
            if prop_vals:
                data[qid] = prop_vals
    return data


  def _level_to_dictionary(self, level):

    # Function that creates a dictionary-like structure for the graph

    graph_dictionary = {}

    for item in level:
      if(item not in graph_dictionary):

        properties = set()
        entities = set()
        item_key = level[item]

        for prop in item_key:

          prop_value  = item_key[prop]
          properties.add(prop)
          if(isinstance(prop_value[0], dict) and 'id' in prop_value[0]):
            entities.add(prop_value[0]['id'])

        graph_dictionary[item] = {'properties': properties, 'entities': entities}

    return graph_dictionary


  def _merge_dictionaries(self, level1, level2):
    # Function that merges the level dictionaries
    merged_dictionary = {}

    for item in level1:

      entities = level1[item]["entities"].copy()
      properties = level1[item]["properties"].copy()

      for entity in level1[item]["entities"]:
        # access level 2 entities
        if(entity in level2):
          if(entity[0] == "Q"):
            entities = level2[entity]["entities"].union(entities)
          properties = level2[entity]["properties"].union(properties)

      merged_dictionary[item] = {"properties": list(properties), "entities": list(entities)}

    for item in merged_dictionary:
      entities = merged_dictionary[item]["entities"].copy()
      properties = merged_dictionary[item]["properties"].copy()

      merged_dictionary[item]["entities"] = [entity for entity in entities if entity[0] == "Q"]

      # Keep only valid properties
      merged_dictionary[item]["properties"] = [prop for prop in properties if prop[0] == "P"]
    return merged_dictionary