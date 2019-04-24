from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import data
import torch
import argparse
import io

from models import BoW, LSTM_encoder, biLSTM_maxp_encoder


# Create dictionary
def create_dictionary(sentences, threshold=0):
	words = {}
	for s in sentences:
		for word in s:
			words[word] = words.get(word, 0) + 1

	if threshold > 0:
		newwords = {}
		for word in words:
			if words[word] >= threshold:
				newwords[word] = words[word]
		words = newwords
	words['<s>'] = 1e9 + 4
	words['</s>'] = 1e9 + 3
	words['<p>'] = 1e9 + 2

	sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
	id2word = []
	word2id = {}
	for i, (w, _) in enumerate(sorted_words):
		id2word.append(w)
		word2id[w] = i

	return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
	word_vec = {}

	with io.open(path_to_vec, 'r', encoding='utf-8') as f:
		# if word2vec or fasttext file : skip first line "next(f)"
		for line in f:
			word, vec = line.split(' ', 1)
			if word in word2id:
				word_vec[word] = np.fromstring(vec, sep=' ')

	logging.info('Found {0} words with word vectors, out of \
		{1} words'.format(len(word_vec), len(word2id)))
	return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
	_, params.word2id = create_dictionary(samples)
	params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
	params.wvec_dim = 300
	return

def batcher(params, batch):
	batch = [sent if sent != [] else ['.'] for sent in batch]

	batch_size = len(batch)

	# get lengths
	lengths = np.zeros(batch_size)
	for i, sent in enumerate(batch):
		lengths[i] = len(sent)

	max_length = int(np.max(lengths))

	sentvecs = np.zeros([max_length, batch_size, emb_dim])

	for i, sent in enumerate(batch):
		for j, word in enumerate(sent):
			if word in params.word_vec:
				sentvecs[j, i, :] = params.word_vec[word]

	with torch.no_grad():

		sentvecs = torch.tensor(sentvecs, dtype = torch.float32).to(device)
		lengths = torch.tensor(lengths, dtype = torch.long).to(device)

		embeddings = encoder_model.forward(sentvecs, lengths)

		if device == "cpu":
			embeddings = embeddings.numpy()
		else:
			embeddings = embeddings.cpu().numpy()

	return embeddings



# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

	global device, emb_dim, PATH_TO_VEC
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	parser = argparse.ArgumentParser()

	parser.add_argument('--checkpoint_path', type = str, default = "./../outputs/biLSTM_maxp/biLSTM_maxp_Adam_X/biLSTM_maxp_bestEncoder.pwf")
	parser.add_argument('--task', type = str, default = "all")
	parser.add_argument('--encoder_name', type = str, choices = ["BoW", "LSTM", "biLSTM", "biLSTM_maxp"], default = "biLSTM")
	parser.add_argument('--path_to_vec', type = str, default = './../SentEval-master/pretrained/glove.840B.300d.txt')
	parser.add_argument('--path_to_senteval', type = str, default = "./../SentEval-master/")
	args, unparsed = parser.parse_known_args()

	emb_dim = 300
	lstm_hidden_size = 2048
	lstm_num_layers = 1
	lstm_dropout_rate = 0
	batch_size = 64
	PATH_TO_VEC = args.path_to_vec
	PATH_TO_SENTEVAL = args.path_to_senteval
	PATH_TO_DATA = PATH_TO_SENTEVAL + "/data/"

	# import SentEval
	sys.path.insert(0, PATH_TO_SENTEVAL)
	import senteval

	# Set params for SentEval
	params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
	params_senteval['classifier'] = {'nhid': 1, 'optim': 'rmsprop', 'batch_size': 128,
									 'tenacity': 3, 'epoch_size': 5}

	args.checkpoint_path = "./../outputs/{}/{}_Adam_X/{}_bestEncoder.pwf".format(args.encoder_name, args.encoder_name, args.encoder_name)

	if args.encoder_name == "BoW":
		encoder_model = BoW()
	elif args.encoder_name == "LSTM":
		encoder_model = LSTM_encoder(False, emb_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout_rate)
	elif args.encoder_name == "biLSTM":
		encoder_model = LSTM_encoder(True, emb_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout_rate)
	elif args.encoder_name == "biLSTM_maxp":
		encoder_model = biLSTM_maxp_encoder(lstm_hidden_size, batch_size, emb_dim, lstm_num_layers, lstm_dropout_rate)

	if args.encoder_name != "BoW":
		encoder_model.load_state_dict(torch.load(args.checkpoint_path, map_location = device))
		encoder_model.to(device)
		encoder_model.eval()

	print("Loaded {} encoder from checkpoint {}".format(args.encoder_name, args.checkpoint_path))

	se = senteval.engine.SE(params_senteval, batcher, prepare)

	if args.task == "all":
		# tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
		# 		  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
		# 		  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
		# 		  'Length', 'WordContent', 'Depth', 'TopConstituents',
		# 		  'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
		# 		  'OddManOut', 'CoordinationInversion']
		tasks = ["MR", "CR", "SUBJ", "TREC", "MRPC", "SICKEntailment"]
	else:
		tasks = args.task

	results = se.eval(tasks)
	print(results)