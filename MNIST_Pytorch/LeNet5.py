"""
filename: LeNet5.py

@author:Su_Chi
@date:2019/9/10 20:59
"""

import torch, torchvision
import argparse
from torch import nn, optim
from torchvision import transforms
from collections import OrderedDict
from functools import reduce
from data_load import data_load
from random import randint

def compose(*funcs):
	"""
	Compose arbitrarily many functions, evaluated left to right.

	Reference:https://mathieularose.com/function-composition-in-python/
	"""
	if funcs:
		return reduce(lambda f,g:lambda *a, **kw: g(f(*a, **kw)), funcs)
	else:
		raise ValueError('Composition of empty sequence not supported.')

class LeNet5(nn.Module):
	def __init__(self, num_classes):
		super(LeNet5, self).__init__()
		self.features = nn.Sequential(OrderedDict([
							('conv_1', self._conv2d(1, 32, 1)),
							('relu_1', self._relu()),
							('bn_1', self._bn(32)),
							('maxpool_1', self._pool()),
							('dropout_1', self._dropout()),
							('conv_2', self._conv2d(32, 64, 1)),
							('relu_2', self._relu()),
							('bn_2', self._bn(64)),
							('maxpool_2', self._pool()),
							('dropout_2', self._dropout()),
							('conv_3', self._conv2d(64, 128, 1)),
							('relu_3', self._relu()),
							('bn_3', self._bn(128)),
							('maxpool_3', nn.MaxPool2d(2, ceil_mode=True)) #Note that floor((7-2)/2+1)=3 thus 'ceil' utilized.
							]))
		self.classifier = nn.Sequential(OrderedDict([
							('dropout_3', self._dropout()),
							('fc_4', nn.Linear(128*4*4, 256, bias=False)),
							('relu_4', self._relu()),
							('dropout_4', self._dropout()),
							('fc_5', nn.Linear(256, num_classes, bias=False))
							]))

	def _conv2d(self, in_channel, out_channel, padding=0, kernel_size=3, bias_bool=False):
		return nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, bias=bias_bool)

	def _linear(self, in_features, out_features, bias_bool= False):
		return nn.Linear(in_features, out_features, bias=bias_bool)

	def _relu(self):
		return nn.ReLU(inplace= True)

	def _bn(self, num_features):
		return nn.BatchNorm2d(num_features)

	def _pool(self):
		return nn.MaxPool2d(2)

	def _dropout(self, pro= 0.5):
		return nn.Dropout(p= pro)

	def forward(self, x):
		x = self.features(x)
		x = x.view(int(x.size()[0]),-1)
		x = self.classifier(x)
		return x

def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss= nn.CrossEntropyLoss()(output, target)
		loss.backward()	 #backward-prop
		optimizer.step() #optimization
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{}] ({:.0f}%)\t loss: {}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset), 
				100.* batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
	model.eval() #dropout and batch_norm fixed
	test_loss=0
	correct=0
	with torch.no_grad():
		for data,target in test_loader:
			data, target= data.to(device), target.to(device)
			output = model(data)
			test_loss += nn.CrossEntropyLoss(reduce=True)(output, target).item()
			"""
			output = nn.LogSoftmax(output)
			pred = output.argmax()
			"""
			pred = output.argmax(1, keepdim=True)#keepdim=True means pred.shape= (num_classes,1)
			correct += torch.sum((pred == target.view_as(pred))).item()
	test_loss = test_loss / len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))

def _main():
	parser = argparse.ArgumentParser(description='MNIST-LeNet5')
	parser.add_argument('-bs', '--batch_size', type=int, default=100, metavar='N',
						help='input batch size for training (default=100)')
	parser.add_argument('-tbs', '--test_batch_size', type=int, default=1000, metavar='N',
						help='input test batch size for testing (default=1000)')
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default= 0.001)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N', 
						help='number of epochs to train (default=10)')
	parser.add_argument('--no_cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default= 1)')
	parser.add_argument('--log_interval', type=int, default=10, metavar='N',
						help='how many epochs to wait before logging the training status')
	parser.add_argument('--save_model', action='store_true', default=False, 
						help='For saving the current model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	#setting the args.seed
	torch.manual_seed(args.seed)
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	
	trainset, testset = data_load(False)
	train_loader = torch.utils.data.DataLoader(trainset, 
											batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(testset, 
											batch_size= args.test_batch_size, shuffle=False, **kwargs)

	lenet = LeNet5(10)#number '10' corresponds to classes in MNIST
	optimizer = optim.RMSprop(lenet.parameters(), lr=args.lr, alpha=0.9)
	if use_cuda:
		lenet.to(device)#move from cpu to gpu

	for epoch in range(1, args.epochs+1):
		train(args, lenet, device, train_loader, optimizer, epoch)
		test(args, lenet, device, test_loader)

	if args.save_model:
		torch.save(lenet.state_dict(), 'MNIST_LeNet5.pt')

def _predict():
	trainset, testset = data_load(False)
	rand = randint(0,len(testset)-1)
	data, target = testset[rand]
	
	lenet = LeNet5(10)
	state_dict = torch.load('MNIST_LeNet5.pt')
	lenet.load_state_dict(state_dict)
	lenet.eval()
	with torch.no_grad():
		output= lenet(data.unsqueeze(0))#data.shape= (1, 28, 28) which input need 4-D.
		pred = output.argmax(1)

	print('Target: {} Predict: {}'.format(
									target.item(), pred.item()))
	
	
if __name__ == '__main__':
	#_main()
	_predict()


