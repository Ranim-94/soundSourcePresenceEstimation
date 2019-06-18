import torch
import torch.nn as nn

class View(nn.Module):
	def __init__(self, *shape):
		super(View, self).__init__()
		self.shape = shape
	def forward(self, input):
		return input.view(input.shape[0], *self.shape)

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
	def forward(self, input):
		return input.view(input.shape[0], -1)

class PresPredCNN(nn.Module):
	def __init__(self):
		super(PresPredCNN, self).__init__()

		self.layers = []
		self.layers.append(nn.Conv2d(1, 128, (5, 5), stride=1, padding=(2, 2), dilation=1, groups=1, bias=True))
		self.layers.append(nn.LeakyReLU())
		self.layers.append(nn.Conv2d(128, 64, (5, 5), stride=1, padding=(2, 2), dilation=1, groups=1, bias=True))
		self.layers.append(nn.LeakyReLU())
		self.layers.append(nn.Conv2d(64, 32, (5, 5), stride=1, padding=(2, 2), dilation=1, groups=1, bias=True))
		self.layers.append(nn.LeakyReLU())
		self.layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=(2, 2), dilation=1, groups=1, bias=True))
		self.layers.append(nn.LeakyReLU())
		self.layers.append(Flatten())
		self.layers.append(nn.Linear(8*29*32, 3))
		self.main = nn.Sequential(*self.layers)

	def forward(self, x):
		return self.main(x)
