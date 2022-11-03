import torch
import torch.nn as nn 
import torch.nn.functional as F




class MultiLayerPerceptron(nn.Module):
  def __init__(self, input_size=784, output_size=10, hidden1 = 64, hidden2 = 64):
    super().__init__()
    self.d1 = nn.Linear(input_size, hidden1)
    self.d2 = nn.Linear(hidden1, hidden2)
    self.d3 = nn.Linear(hidden2, output_size)
    return
  
  def forward(self, X):
    X = self.d1(X)
    X = F.relu(X)
    X = self.d2(X)
    X = F.relu(X)
    X = self.d3(X)
    return F.log_softmax(X, dim=1)