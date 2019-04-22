import torch
from torch import nn
import numpy as np
import sys

class BoW(object):
	def forward(self, x, l):

		# sum along sentence length dimension
		emb_sum = x.sum(0)

		# divide with the size of each sentence
		emb = emb_sum / l.float().view([-1, 1])

		return emb


class LSTM_encoder(nn.Module):
	def __init__(self, bidirectionality, emb_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout_rate):
		super(LSTM_encoder, self).__init__()

		self.bidirectionality = bidirectionality

		self.encoder_layer = nn.LSTM(input_size = emb_dim, hidden_size = lstm_hidden_size, num_layers = lstm_num_layers,
									 bidirectional = self.bidirectionality, dropout = lstm_dropout_rate)

	def forward(self, x, l):

		# get sorted index for the lengths (descending order)
		idx_sorted = torch.argsort(l, descending = True)

		# sort lengths
		l_sorted = torch.index_select(l, dim = 0, index = idx_sorted)

		# sort batch (along batch dimension) based on the lengths
		x_sorted = torch.index_select(x, dim = 1, index = idx_sorted)

		# pad batch
		x_sorted_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, l_sorted)

		# get hidden state of last layer
		emb_sorted = self.encoder_layer(x_sorted_packed)[1][0] # [num_layers * num_directions, batch_size, hidden_size]
			
		# concatinate the hidden state from the two directions
		if self.bidirectionality:
			emb_sorted = torch.cat([emb_sorted[0], emb_sorted[1]], 1) # [batch_size, 2 * hidden_size]

		# otherwise just get rid of first dimension
		else:
			emb_sorted = emb_sorted.squeeze(0) # [batch_size, hidden_size]

		# get index to restore original order of the batch
		idx_unsorted = torch.argsort(idx_sorted, descending = False)

		# unsort batch
		emb = torch.index_select(emb_sorted, dim = 0, index = idx_unsorted)

		return emb


class biLSTM_maxp_encoder(nn.Module):
	def __init__(self, lstm_hidden_size, batch_size, emb_dim, lstm_num_layers, lstm_dropout_rate):
		super(biLSTM_maxp_encoder, self).__init__()

		self.lstm_hidden_size = lstm_hidden_size
		self.batch_size = batch_size

		self.encoder_layer = nn.LSTM(input_size = emb_dim, hidden_size = lstm_hidden_size, num_layers = lstm_num_layers,
								     bidirectional = True, dropout = lstm_dropout_rate)

		self.linear_projection = nn.Linear(2 * lstm_hidden_size, 2 * lstm_hidden_size, bias = False)

	def forward(self, x, l):

		# get sorted index for the lengths (descending order)
		idx_sorted = torch.argsort(l, descending = True)

		# sort lengths
		l_sorted = torch.index_select(l, dim = 0, index = idx_sorted)

		# sort batch (along batch dimension) based on the lengths
		x_sorted = torch.index_select(x, dim = 1, index = idx_sorted)

		# pad batch
		x_sorted_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, l_sorted)

		# get output at every time-step
		emb_sorted = self.encoder_layer(x_sorted_packed)[0] # [seq_len, batch_size, 2 * hidden_size]

		# opposite opearation of pack_padded (returns tuple, we only need the first one)
		emb_sorted = nn.utils.rnn.pad_packed_sequence(emb_sorted)[0] # [seq_len, batch_size, 2 * hidden_size]

		# get index to restore original order of the batch
		idx_unsorted = torch.argsort(idx_sorted, descending = False)

		# unsort batch
		emb = torch.index_select(emb_sorted, dim = 1, index = idx_unsorted)

		# 1. reshape to [seq_len * batch_size, 2 * hidden_size]
		# 2. linear transformation (exact same dimensions)
		# 3. reshape back to original shape [seq_len, batch_size, 2 * hidden_size]
		this_batch_size = emb.shape[1]

		emb_proj = self.linear_projection(emb.view(-1, 2*self.lstm_hidden_size)).view(-1, this_batch_size, 2*self.lstm_hidden_size)

		# max pooling on firt dimension (then get rid of first dimension)
		# view is used to accout for infer.py where we use 1 batch
		emb_proj_maxp = torch.max(emb_proj, 0)[0].squeeze(0).view([this_batch_size, 2*self.lstm_hidden_size]) # [batch_size, 2 * hidden_size]

		return emb_proj_maxp


class MLP(nn.Module):
	def __init__(self, args):
		super(MLP, self).__init__()

		self.hidden_sizes = args.classifier_hidden_sizes
		self.actv_fun = args.classifier_actv_fun
		self.dropout_rate = args.classifier_dropout_rate
		self.batch_norm = args.classifier_batch_norm
		self.num_classes = args.num_classes
		self.encoder_name = args.encoder

		assert self.encoder_name in ["BoW", "LSTM", "backwardLSTM", "biLSTM", "biLSTM_maxp", "biLSTM_minmax"]
		assert self.actv_fun in ["ReLU", "tanh", "linear"]

		if self.encoder_name == "BoW":
			self.n_dim = 4 * args.emb_dim
			self.encoder = BoW()

		elif self.encoder_name == "LSTM":
			self.n_dim = 4 * args.lstm_hidden_size
			self.encoder = LSTM_encoder(False, args.emb_dim, args.lstm_hidden_size, args.lstm_num_layers, args.lstm_dropout_rate)

		elif self.encoder_name == "biLSTM":
			self.n_dim = 4 * 2 * args.lstm_hidden_size
			self.encoder = LSTM_encoder(True, args.emb_dim, args.lstm_hidden_size, args.lstm_num_layers, args.lstm_dropout_rate)

		elif self.encoder_name == "biLSTM_maxp":
			self.n_dim = 4 * 2 * args.lstm_hidden_size
			self.encoder = biLSTM_maxp_encoder(args.lstm_hidden_size, args.batch_size, args.emb_dim, args.lstm_num_layers, args.lstm_dropout_rate)

		modules = []

		self.hidden_sizes = [self.n_dim] + self.hidden_sizes
		n_layers = len(self.hidden_sizes)

		for i in range(n_layers - 1):

			modules.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))

			# Activation layer
			if self.actv_fun == "ReLU":
				modules.append(nn.ReLU())
			elif self.actv_fun == "tanh":
				modules.append(nn.tanh())

			if self.dropout_rate:
				modules.append(nn.Dropout(p = self.dropout_rate))

			if self.batch_norm:
				modules.append(nn.BatchNorm1d(self.hidden_sizes[i + 1]))

		modules.append(nn.Linear(self.hidden_sizes[-1], self.num_classes))

		self.layers = nn.Sequential(*modules)

	def forward(self, x1, x2, len1, len2):

		# encode sentence batches
		emb1 = self.encoder.forward(x1, len1)
		emb2 = self.encoder.forward(x2, len2)

		# concatenate bacthed embeddings along second dimension
		concatenated_embeddings = torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], 1)

		# get logits from classification model
		logits = self.layers(concatenated_embeddings)

		return logits
