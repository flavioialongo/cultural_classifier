# predictor.py
import torch 
from abc import ABC, abstractmethod

from datasets import Dataset as HFaceDataset 
import pandas as pd
from tqdm import tqdm 


class Predictor(ABC):

  _shared_wikidict = None

  def __init__(self, dataset, data_folder, wikidict=None):
      from wikitext_creator import WikitextCreator
      self.dataset = dataset

      if(isinstance(dataset, HFaceDataset)):
          self.is_pandas = False
      else:
          self.is_pandas = True

      self.data_folder = data_folder

      # Wikidict is used for both models
      # Graph is created only when needed (predict_textgraph)
      if wikidict is not None:
          self.wikidict = wikidict
          Predictor._shared_wikidict = wikidict
      elif Predictor._shared_wikidict is not None:
          print("Reusing Wikipedia dictionary")
          self.wikidict = Predictor._shared_wikidict
      else:
          print("Creating Wikipedia articles dictionary")
          self.wikidict = WikitextCreator(dataset).wikidict
          Predictor._shared_wikidict = self.wikidict

  @abstractmethod
  def predict(self, save_path):
      pass

class TransformerPredictor(Predictor):
  def __init__(self, dataset, data_folder, wikidict=None):

      from transformers import (
          AutoModelForSequenceClassification,
          AutoTokenizer,
      )

      super().__init__(dataset=dataset, data_folder=data_folder, wikidict = wikidict)
      
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/NLP_HOMEWORK_1/SocioEmbeddings/data/transformer/", local_files_only = True)
      self.tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/NLP_HOMEWORK_1/SocioEmbeddings/data/transformer/", local_files_only = True)
      
      self.model.to(self.device)

  def predict(self, save_path):

      def predict_culturality(text, model, tokenizer, device, max_length=512):
          """
              Sub-function that predicts the culturality for a given text
          """
          model.eval()
          encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
          input_ids = encoding['input_ids'].to(device)
          attention_mask = encoding['attention_mask'].to(device)
          with torch.no_grad():
              outputs = model(input_ids=input_ids, attention_mask=attention_mask)
              _, preds = torch.max(outputs.logits, dim=1)

          if preds.item() == 0:
              return "cultural exclusive"
          elif preds.item() == 1:
              return "cultural agnostic"
          else:
              return "cultural representative"

      results = []

      for idx in tqdm(range(len(self.dataset))):
          if(self.is_pandas):
              row = self.dataset.iloc[idx]
          else:
              row = self.dataset[idx]

          qid = row["item"].split("/")[-1]

          text = self.prompt(row, self.wikidict[qid])

          # Make prediction based on description text
          predicted_label = predict_culturality(text, self.model, self.tokenizer, self.device)

          # Append results with all original columns + predicted label
          results.append({
              "item": row["item"],
              "name": row["name"],
              "description": row["description"],
              "type": row["type"],
              "category": row["category"],
              "subcategory": row["subcategory"],
              "label": predicted_label
          })

      # Convert results to DataFrame
      results_df = pd.DataFrame(results)

      # Save it to CSV
      results_df.to_csv(save_path, index=False)
      
      print(f"\nPredictions saved to {save_path}")
      return results_df
  def prompt(self, row, text):

      item_name = row["name"]
      description = row["description"]
      item_type = row["type"]
      category = row["category"]
      article = text 

      return f"""
            Task: You are given a cultural item. Classify it into one of the three categories: 'exclusive', 'agnostic', or 'representative'.

            Definitions:
            Cultural Exclusive: The item is known or used only within a specific culture and is not widely recognized outside of it.
            Cultural Agnostic: The item is commonly known or used worldwide, without strong association to any particular culture.
            Cultural Representative: The item originated in a specific culture and is culturally claimed, but it is also known and used across other cultures.

            Instructions:
            Carefully read the information provided below. Based on the definitions above, assign the most appropriate label to the item.

            Item: {item_name}
            Description: {description}
            Type: {item_type}
            Category: {category}

            Full text: {article}
            """



class NonLMPredictor(Predictor):
  def __init__(self, dataset, data_folder, wikidict=None):
      from torch.utils.data.dataloader import DataLoader
      from graph_creator import GraphCreator
      from wikigraph_dataset import WikiGraphDataset
      from cultural_classifier import CulturalClassifier
      super().__init__(dataset=dataset, data_folder=data_folder, wikidict=wikidict)
      
      print("Creating Wikidata graph structure")
      self.wikidata_graph = GraphCreator(self.dataset).graph
      print("Preprocessing data")
    
      self.wikigraph = WikiGraphDataset(dataset=self.dataset,
                                      is_test=True,
                                      data_folder=self.data_folder,
                                      graph=self.wikidata_graph,
                                      wikidict=self.wikidict)
      
      self.dataloader = DataLoader(self.wikigraph, batch_size = 32, collate_fn = self.wikigraph.collate_fn)

      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      # Retrieve pre-trained model

      print("Retrieving the model")
      self.cultural_classifier = CulturalClassifier(self.wikigraph.w2v_embedding, self.wikigraph.graph_embedding, 300)
      self.cultural_classifier.load_state_dict(torch.load(self.data_folder+"text_graph/classifier.pt", weights_only=True, map_location=self.device))

      self.cultural_classifier.to(self.device)



  def predict(self, save_path):
      
      all_predictions = []
      for batch in self.dataloader:

        input = batch['wiki'].to(self.device)
        graph = batch['graph'].to(self.device)

        # No need to track gradients for prediction
        with torch.no_grad():
          outputs = self.cultural_classifier(input, graph)

          probabilities = torch.softmax(outputs, dim=1)
          predicted_classes = torch.argmax(probabilities, dim=1)

          all_predictions.extend(predicted_classes.cpu().numpy())

      results = []

      for idx in tqdm(range(len(self.dataset))):

        if(self.is_pandas):
          row = self.dataset.iloc[idx]
        else:
          row = self.dataset[idx]

        id2label = {
            0: "cultural exclusive",
            1: "cultural agnostic",
            2: "cultural representative"
        }
        # Append results with all original columns + predicted label
        results.append({
            "item": row["item"],
            "name": row["name"],
            "description": row["description"],
            "type": row["type"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "label": id2label[all_predictions[idx]]
        })

      # Convert results to DataFrame
      results_df = pd.DataFrame(results)

      # Save it to CSV
      results_df.to_csv(save_path, index=False)

      print(f"\nPredictions saved to {save_path}")
      return results_df

class PredictorCreator:
  @staticmethod
  def create(model_type: str, dataset, data_folder: str, wikidict=None):
      if model_type=="lm":
          return TransformerPredictor(dataset=dataset, data_folder = data_folder, wikidict=wikidict)
      elif model_type=="nonlm":
          return NonLMPredictor(dataset=dataset, data_folder = data_folder, wikidict=wikidict)
      else:
          raise Exception("Invalid value for model_type, expecting 'lm' or 'nonlm'")
class Predict:
  def __init__(self, model_type: str, dataset, data_folder: str, wikidict=None):
      self.predictor = PredictorCreator.create(model_type = model_type, dataset = dataset, data_folder = data_folder, wikidict=wikidict)

  def predict(self, save_path):
      return self.predictor.predict(save_path)
