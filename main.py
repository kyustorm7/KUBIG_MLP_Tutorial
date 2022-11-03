import torch
import torch.nn as nn
import torch.nn.functional as F

from network import MultiLayerPerceptron
from loader.dataloader import train_loader, test_loader

device = torch.device("cuda")

criterion = nn.CrossEntropyLoss()
criterion.to(device)

MLP = MultiLayerPerceptron()
MLP.to(device)


optimizer = torch.optim.Adam(MLP.parameters(), lr = 0.001)

epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for epoch in range(epochs):
  trn_corr = 0
  tst_corr = 0

  for number, (image, y_true) in enumerate(train_loader):
    image = image.to(device)
    y_true = y_true.to(device)

    number += 1
    # image n * 1 * 28 * 28
    image = image.view(-1, 784)
    y_pred = MLP(image)
    loss = criterion(y_pred, y_true)

    predicted = torch.max(y_pred.data, 1)[1]
    batch_corr = (predicted == y_true).sum()
    trn_corr += batch_corr

    # Update parameters
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    # Print Current Loss
    if number%500 ==0:
      print(f'epoch {epoch:2} batch:{number:4} Train loss: {loss.item(): 10.8f} Train accuracy: {trn_corr.item(): 10.8f}')
    
  train_losses.append(loss)
  train_correct.append(trn_corr)

  with torch.no_grad():
    for number, (image, y_true) in enumerate(test_loader):
      image, y_true = image.to(device), y_true.to(device)
      image = image.view(-1, 784)
      y_val = MLP(image)
      predicted = torch.max(y_val.data, 1)[1]
      tst_corr += (predicted == y_true).sum()

    loss = criterion(y_val, y_true)
    test_losses.append(loss)
    test_correct.append(tst_corr)

