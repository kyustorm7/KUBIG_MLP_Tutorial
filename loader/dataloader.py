from PIL import Image
import torch 
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataload_root = "data"
mnist_transform = transforms.ToTensor()

train_dataset = datasets.MNIST(dataload_root, train = True, download=True, transform = mnist_transform)
test_dataset = datasets.MNIST(dataload_root, train = False, download=True, transform = mnist_transform)

train_loader = DataLoader(train_dataset, batch_size = 64)
test_loader = DataLoader(test_dataset, batch_size= 64)

