import time
from model import *
import librosa as lr
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import torch.optim as optim
from data_loader import *
from training import PresPredTrainer


def main(config):
	train_dataset = PresPredDataset('dev', train_data_path='data/train/', sampling_rate=32000, val_split=0.3)
	test_dataset = PresPredDataset('eval', train_data_path='data/test/', sampling_rate=32000)
	test_recrep_dataset = PresPredDatasetSimple('test_recrep', train_data_path='data/test_recrep/', sampling_rate=32000)
	test_sim_dataset = PresPredDatasetSimple('test_sim', train_data_path='data/test_sim/', sampling_rate=32000)
	model = PresPredCNN()
	
	# Model est. presence
	model = torch.load('save/model_2019-02-21_16-12-30')
	
	model.cuda()
	trainer = PresPredTrainer(model, train_dataset, test_dataset=test_dataset, optimizer=optim.Adam, lr=config.lr, loss_fun=config.loss)
	trainer.train(batch_size=config.batch_size, epochs=config.epochs)
	trainer.test(batch_size=43)
	trainer.test_simple(test_recrep_dataset, 'test_recrep', batch_size=43)
	trainer.test_simple(test_sim_dataset, 'test_sim', batch_size=43)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    
	# Training
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=10)
	
	config = parser.parse_args()
	main(config)
