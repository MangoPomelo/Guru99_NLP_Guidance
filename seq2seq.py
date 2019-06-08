import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import os
import re
import random
# http://www.manythings.org/anki/
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
	def __init__(self):
		#initialize containers to hold the words and corresponding index
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS
	
	def add_sentence(self, sentence):
		#split a sentence into words and add it to the container
		for word in sentence.split(' '):
			self.add_word(word)
	
	def add_word(self, word):
		#If the word is not in the container, the word will be added to it, 
		#else, update the word counter
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

def process_data(file_path):
	df = pd.read_csv(file_path, delimiter = '\t', header = None, names = ['English', 'Indonesian'])
	def normalize(sentence):
		sentence = sentence.lower()
		sentence = re.sub('[^A-Za-z\s]+', '', sentence)
		sentence = sentence.encode('ascii', errors='ignore').decode('utf-8')
		return sentence
	df['English'], df['Indonesian'] = df['English'].apply(normalize), df['Indonesian'].apply(normalize)
	source, target = Lang(), Lang()
	pairs = []
	for i in range(len(df)):
		if len(df['English'][i].split()) < MAX_LENGTH and len(df['Indonesian'][i].split()) < MAX_LENGTH:
			full = [df['English'][i], df['Indonesian'][i]]
			source.add_sentence(df['English'][i]); target.add_sentence(df['Indonesian'][i])
			pairs.append(full)
	return source, target, list(pairs)

def tensor_from_sentence(lang, sentence):
	return torch.tensor([lang.word2index[word] for word in sentence.split()] + [EOS_token], dtype = torch.long, device = device).reshape(-1, 1)

def tensor_from_pair(input_lang, output_lang, pair):	
	return (tensor_from_sentence(input_lang, pair[0]), tensor_from_sentence(output_lang, pair[1]))

class Encoder(nn.Module):
	def __init__(self, input_size, hidden_dim, embed_dim, num_layers):
		super(Encoder, self).__init__()
		self.input_size = input_size
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.embedding = nn.Embedding(input_size, embed_dim)
		self.gru = nn.GRU(embed_dim, hidden_dim, num_layers = num_layers)
	def forward(self, x):
		# Input single word with shape [1] dim = 1
		embedded = self.embedding(x).reshape(1, 1, -1)
		# Here, inflate to dim = 3,which means (BATCH_SZIE = 1, TIME_STEP = 1, FEATURE = 256)
		outputs, h_n = self.gru(embedded)
		return outputs, h_n

class Decoder(nn.Module):
	def __init__(self, input_size, hidden_dim, embed_dim, num_layers):
		super(Decoder, self).__init__()
		self.input_size = input_size
		self.hidden_dim = hidden_dim
		self.embed_dim = embed_dim
		self.num_layers = num_layers
		self.embedding = nn. Embedding(input_size, embed_dim)
		self.gru = nn.GRU(embed_dim, hidden_dim, num_layers = num_layers)
		self.linear = nn.Linear(hidden_dim, input_size)
	def forward(self, x, h_n):
		# single word inputting, shape [1]
		embedded = torch.relu(self.embedding(x).reshape(1, 1, -1)) # identical trick as encoder does
		output, h_n = self.gru(embedded, h_n)
		prediction = torch.log_softmax(self.linear(output[0]), dim = 1)
		# output[0].shape = (1, HIDDEN_SIZE), therefore dim = 1
		return prediction, h_n

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, device):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.device = device
	def forward(self, source, target, teacher_forcing_ratio=0.5):
		source_length = source.shape[0] # timestep
		target_length = target.shape[0]

		feature_size = target.shape[1]
		vocab_size = self.decoder.input_size
	  
		#initialize a variable to hold the predicted outputs
		outputs = torch.zeros(target_length, feature_size, vocab_size).to(self.device)

		#encode every word in a sentence
		for i in range(source_length):
			encoder_output, encoder_hidden = self.encoder(source[i])

		#use the encoder’s last hidden layer as the decoder hidden
		decoder_hidden = encoder_hidden.to(device)
  
		#add a token before the first predicted word
		decoder_input = torch.tensor([SOS_token], device=device)  # SOS

		#topk is used to get the top K value over a list
		#predict the output word from the current target word. If we enable the teaching force,  then the
		#next decoder input is the next word, else, use the decoder output highest value. 

		for t in range(target_length):   
			decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
			outputs[t] = decoder_output
			teacher_force = random.random() < teacher_forcing_ratio
			topv, topi = decoder_output.topk(1)
			decoder_input = (target[t] if teacher_force else topi) # topi actually is the word_index
			if(teacher_force == False and decoder_input.item() == EOS_token):
				break

		return outputs

teacher_forcing_ratio = 0.5

def calcModel(model, input_tensor, target_tensor, model_optimizer, criterion):
	model_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	loss = 0
	epoch_loss = 0
	# print(input_tensor.shape)

	output = model(input_tensor, target_tensor)

	num_iter = output.size(0) # timestep of target sentence

	#calculate the loss from a predicted sentence with the expected result
	for ot in range(num_iter):
		loss += criterion(output[ot], target_tensor[ot])

	loss.backward()
	model_optimizer.step()
	epoch_loss = loss.item() / num_iter

	return epoch_loss

def trainModel(model, source, target, pairs, num_iteration=20000):
	model.train()

	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	criterion = nn.NLLLoss()
	total_loss_iterations = 0

	training_pairs = [tensor_from_pair(source, target, random.choice(pairs)) for i in range(num_iteration)]
  
	for iter in range(1, num_iteration+1):
		training_pair = training_pairs[iter - 1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = calcModel(model, input_tensor, target_tensor, optimizer, criterion)

		total_loss_iterations += loss

		if iter % 5000 == 0:
			avarage_loss= total_loss_iterations / 5000
			total_loss_iterations = 0
			print('%d %.4f' % (iter, avarage_loss))
		  
	torch.save(model.state_dict(), 'mytraining.pt')
	return model

def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
	with torch.no_grad():
		input_tensor = tensor_from_sentence(input_lang, sentences[0])
		output_tensor = tensor_from_sentence(output_lang, sentences[1])

		decoded_words = []

		output = model(input_tensor, output_tensor) # shape (target_len, feature_size, vocab_size)
		# print(output_tensor)

		for ot in range(output.size(0)):
			topv, topi = output[ot].topk(1)
			# print(topi)

			if topi[0].item() == EOS_token:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(output_lang.index2word[topi[0].item()])
	return decoded_words

def evaluateRandomly(model, source, target, pairs, n=10):
	for i in range(n):
		pair = random.choice(pairs)
		print('source {}'.format(pair[0]))
		print('target {}'.format(pair[1]))
		output_words = evaluate(model, source, target, pair)
		output_sentence = ' '.join(output_words)
		print('predicted {}'.format(output_sentence))

source, target, pairs = process_data("./ind.txt")
randomize = random.choice(pairs)
print('random sentence {}'.format(randomize))

#print number of words
INPUT_SIZE = source.n_words
OUTPUT_SIZE = target.n_words
print('Input : {} Output : {}'.format(INPUT_SIZE, OUTPUT_SIZE))

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
NUM_ITERATION = 100000

#create encoder-decoder model
encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, EMBED_SIZE, NUM_LAYERS)
decoder = Decoder(OUTPUT_SIZE, HIDDEN_SIZE, EMBED_SIZE, NUM_LAYERS)

model = Seq2Seq(encoder, decoder, device).to(device)

#print model 
print(encoder)
print(decoder)

model = trainModel(model, source, target, pairs, NUM_ITERATION)
evaluateRandomly(model, source, target, pairs)