MNLP HOMEWORK 1
Lorenzo Gandini – Flavio Ialongo

This is the shared folder containing all files for our project.

Main Notebook
Evaluation.ipynb is the main notebook for running the classification. It supports both solutions:

The "lm" section refers to the transformer-based model.

The "non-lm" section refers to the hybrid approach combining Word2Vec and Node2Vec embeddings.

This notebook depends on the Python scripts located in the utils folder.
All models used are pre-trained and were generated using the corresponding notebooks listed below.

Supporting Notebooks
Hybrid_Solution.ipynb – This notebook is dedicated to our hybrid (non-lm) solution.
Training cells are commented out to avoid overwriting the models used in evaluation.

Transformer.ipynb – This notebook trains the transformer-based solution using microsoft/deberta-v3-base.
Running it will overwrite the current saved model in the corresponding folder.

Graph.ipynb – Demonstrates the construction of the Wikidata graph and the generation of Node2Vec embeddings.

Report
SocioEmbeddings_Report.pdf – Contains our written report describing the methodology, models, and evaluation results.