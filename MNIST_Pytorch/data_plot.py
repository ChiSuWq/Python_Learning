"""
@filename: data_plot.py

@author:Su_Chi
@date: 2019/9/10 19:01
"""

import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image



def data_plot():
	fig, ax = plt.subplots(nrows=2, ncols=5, sharex= True, sharey= True)
	ax = ax.flatten()
	#output number 0-9
	for i in range(10):
		#print(np.argwhere((trainset.train_labels.numpy()==i)==1).shape)
		#print(np.nonzero(trainset.train_labels.numpy()==i)[0][0])
		img = trainset[np.nonzero(trainset.train_labels.numpy()==i)[0][0]][0]
		ax[i].imshow(img, cmap='Greys_r', interpolation='nearest')

	ax[0].set_xticks([])
	ax[0].set_yticks([])
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	trainset = torchvision.datasets.MNIST('./data', train=True)# transform=transform, download=True)
	#trainset contains train_data and train_labels which are both tensors.
	print(trainset.train_data.size())
	#data_plot()


