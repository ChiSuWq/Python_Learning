"""
filename:data_load.py

@author:Su_Chi
@date: 2019/9/10 18:39
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

def data_load(download= True):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))])
	trainset = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=download)
	#trainloader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	testset = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=download)
	#testloader = data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

	return trainset, testset

def data_loader(download= True):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))])
	trainset = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=download)
	trainloader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	testset = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=download)
	testloader = data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

	return trainloader, testloader

if __name__ == '__main__':
	print('Loading data')
	trainloader, testloader = data_loader()
	iter_data, iter_labels = iter(trainloader).next()
	print(iter_labels)
	print('Loading finished')