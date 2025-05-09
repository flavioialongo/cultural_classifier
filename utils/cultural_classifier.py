import torch
import torch.nn as nn
import torch.nn.functional as F

class CulturalClassifier(nn.Module):
    def __init__(self, word_embedding, graph_embedding, embedding_size):
        super().__init__()

        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, padding_idx=0)
        self.graph_embedding = nn.Embedding.from_pretrained(graph_embedding, padding_idx=0)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # for word
        self.beta = nn.Parameter(torch.tensor(0.5))   # for graph


        self.output = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embedding_size, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150,3)
        )

    def forward(self, input, graph):
      """
      input: (batch_size, seq_len)       -- word indices
      graph: (batch_size, num_nodes)     -- graph node indices (with 0 as padding)
      """

      weights = F.softmax(torch.stack([self.alpha, self.beta]), dim=0)
      alpha = weights[0]
      beta = weights[1]


      # Retrieve Embeddings
      word_embeddings = self.word_embedding(input) #(batch, seq_len, emb)
      graph_embs = self.graph_embedding(graph)  #(batch, nodes, emb)



    # Calculate Mean for the Graph Embedding without weighting the padded tokens
      # Mask for padding
      graph_mask = (graph != 0).unsqueeze(-1).float()  #(batch, nodes, 1)  1 if non zero, 0 otherwise

      summed = (graph_embs * graph_mask).sum(dim=1)  #(batch, emb)

      # Count valid (non-zero) values, clamp so that no division by 0
      counts = graph_mask.sum(dim=1).clamp(min=1)    #(batch, 1)

      # Average of graph embedding without counting the padding
      mean_graph_emb = summed / counts #(batch, emb)


    # Calculate Mean for the Word Embedding without weighting the padded tokens

      # Word embedding mask
      word_mask = (input != 0).unsqueeze(-1).float() #(batch, seq_len, 1)

      # Mask word embeddings before combining
      masked_word_embeddings = word_embeddings * word_mask  # (B, L, D)

      #Average of tgext embedding without counting the padding clamp so that no division by 0
      mean_word_emb = (masked_word_embeddings * word_mask).sum(dim=1) / word_mask.sum(dim=1).clamp(min=1)

      # Combine embeddings
      combined = mean_word_emb + mean_graph_emb

      # Output layer
      output = self.output(combined)  # (B, 3)

      return output

