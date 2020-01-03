import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import time
from tqdm import tqdm
from data_loader import fetch_data

w2v = None
device = None
num_total = 0
num_unk = 0
unk = '<UNK>'
rand_vecs = {}

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, batch_size): # Add relevant parameters
		super(RNN, self).__init__()
		# Fill in relevant parameters
		# Ensure parameters are initialized to small values, see PyTorch documentation for guidance
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.rnn = nn.RNN(input_size, hidden_size,)
		self.activation = nn.Tanh()
		self.linear1 = nn.Linear(hidden_size, hidden_size)
		nn.init.uniform_(self.linear1.weight.data, -math.sqrt(6)/math.sqrt(2 * hidden_size), math.sqrt(6)/math.sqrt(2 * hidden_size))
		self.linear2 = nn.Linear(hidden_size, output_size)
		nn.init.uniform_(self.linear2.weight.data, -math.sqrt(6)/math.sqrt(hidden_size + 5), math.sqrt(6)/math.sqrt(hidden_size + 5))
		self.softmax = nn.LogSoftmax(dim=1)
		self.loss = nn.CrossEntropyLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, inputs): 
		#begin code
		h_0 = torch.rand(1,self.batch_size, self.hidden_size, dtype=torch.double).to(device)
		output, hidden = self.rnn(inputs)
		l1 = self.linear1(output[-1])
		l2 = self.linear2(self.activation(l1))
		predicted_vector = self.softmax(l2) 
		# Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
		#end code
		return predicted_vector


# Converts list of pairs (doc, y) into pair of (list of v for each doc, array of y for each doc)
def vectorize_data(data):
	vecs = []
	labels = torch.zeros(len(data))
	for i, (doc, y) in enumerate(data):
		vec = vectorize_doc(doc)
		vecs.append(torch.tensor(vec))
		labels[i] = y
	print (num_total, num_unk)
	return vecs, labels

# Convert doc into v where v has shape (n, d) where n is then number of words in doc and
# d is the number of dimensions in the word embedding
def vectorize_doc(doc):
	res = np.zeros((len(doc), 50))
	for i in range(len(doc)):
		res[i] = vectorize_word(doc[i])
	return res.reshape((len(doc), 1, 50))

# Convert word into v of size d
def vectorize_word(word):
	global num_unk, num_total
	word = word.lower()
	num_total += 1
	if not word in w2v:
		if word in rand_vecs:
			return rand_vecs[word]
		else:
			r = np.random.random(50)
			rand_vecs[word] = r
			num_unk += 1
			return r
	return w2v.word_vec(word)

def main(hidden_dim, batch_size): 
	global device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	if not os.path.exists('glove.6B.50d.w2v.txt'):
		print("w2v file not found, generating...")
		glove2word2vec(glove_input_file='glove.6B.50d.txt', word2vec_output_file='glove.6B.50d.w2v.txt')
	global w2v
	w2v = KeyedVectors.load_word2vec_format('glove.6B.50d.w2v.txt', binary=False)

	print("Fetching data...")
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

	model = RNN(50,hidden_dim,5,batch_size)
	model.double()
	model.cuda()

	print("Vectorizing data...")
	train_vecs, train_labs = vectorize_data(train_data)
	valid_vecs, valid_labs = vectorize_data(valid_data)
	print("Finished vectorizing data")

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=False) 
	#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	iters = 10
	while iters > 0: # How will you decide to stop training and why
		model.train()
		optimizer.zero_grad()
		minibatch_size = 16
		N = len(train_data)
		perm = np.random.permutation(N)
		train_vecs = [train_vecs[i] for i in perm]
		train_labs = train_labs[perm]
		total = 0
		correct = 0
		epoch = 10 - iters
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				gold_label = train_labs[minibatch_index * minibatch_size + example_index].long()
				predicted_vector = model(train_vecs[minibatch_index * minibatch_size + example_index].to(device))
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]).to(device))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size
			loss.backward()
			optimizer.step()

		optimizer.zero_grad() 
		N = len(valid_data)
		total = 0
		correct = 0
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				gold_label = valid_labs[minibatch_index * minibatch_size + example_index].long()
				predicted_vector = model(valid_vecs[minibatch_index * minibatch_size + example_index].to(device))
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		#scheduler.step()
		iters -= 1



