from datasets import Dataset as HFaceDataset 
import pandas as pd
import re 
from tqdm import tqdm 

class WikitextCreator():
  """
    WikitextCreator scrapes the Wikipedia articles relative to the Wikidata entities

    Attributes:
      - dataset: the given dataset
      - wikidict: the dictionary with QID: Wikipedia article
  """
  def __init__(self, dataset):
    from wikidata.client import Client
    self.dataset = dataset

    if(isinstance(dataset, HFaceDataset)):
      self.entities = [x.split("/")[-1] for x in dataset["item"]]

    elif(isinstance(dataset, pd.DataFrame)):
      items = dataset["item"].tolist()  # For pandas DataFrame
      self.entities = [x.split("/")[-1] for x in items]
    else:
      raise Exception("Unknown dataset type")

    self.client = Client()
    self.wikidict = self._retrieve_wikidict()

  def _retrieve_wikidict(self):
    import time

    entities_id = self.entities
    # Wikidata client instantiation
    client = self.client

    # Useful sub-functions
    def clean_wikipedia_extract(text):
        """Sub-function that cleans wikipedia text"""

        # Remove unwanted paragraphs

        text = re.sub(r"^==.*?==\s*", "", text, flags=re.MULTILINE)

        end_markers = ["See also", "References", "External links", "Further reading"]
        for marker in end_markers:
            pattern = rf"==\s*{marker}\s*==.*"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        return text.strip()

    def get_text(item):
      import requests

      """
      Sub-function that handles the get request from Wikipedia

      Arguments:
      item -- wikidata.Entity
      """
      sitelinks = item.data.get("sitelinks", {})
      enwiki = sitelinks.get("enwiki")
      if enwiki:
        title = enwiki["title"]

        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "format": "json",
            "redirects": 1
        }
      
        try:
            res = requests.get(api_url, params=params)
            time.sleep(1)
            res.raise_for_status()
            data = res.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return ""
            page = next(iter(pages.values()))
            text = page.get("extract", "")
            text = text.lower()
            text = clean_wikipedia_extract(text)
            return text
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return ""
        except requests.exceptions.JSONDecodeError:
            print("Response was not valid JSON")
            return ""


    tot = len(entities_id)
    dic = {}


    for entity_id in tqdm(entities_id, total=tot):
      item = client.get(entity_id, load = True)
      text = get_text(item)
      dic[entity_id] = text

    return dic
