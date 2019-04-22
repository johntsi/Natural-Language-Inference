import pandas as pd 
import numpy as np
from ast import literal_eval

class Dataset(object):
	def __init__(self, args):

		s = ""
		if args.lowercase:
			s = "_lower"

		m = ""
		if args.unknown_mapped:
			m = "_mapped"

		# take only part of the sets (for debugging)
		if args.percentage_of_train_data == 100:
			nrows_train = None
		elif args.percentage_of_train_data == 0:
			nrows_train = 1
		else:
			nrows_train = int(args.percentage_of_train_data * len(pd.read_csv(args.data_path + "/snli/train" + s + m + ".txt", sep = "\t", usecols = [0])) / 100)

		if args.percentage_of_dev_data == 100:
			nrows_dev = None
		elif args.percentage_of_dev_data == 0:
			nrows_dev = 1
		else:
			nrows_dev = int(args.percentage_of_dev_data * len(pd.read_csv(args.data_path + "/snli/dev" + s + m + ".txt", sep = "\t", usecols = [0])) / 100)

		if args.percentage_of_test_data == 100:
			nrows_test = None
		else:
			nrows_test = int(args.percentage_of_test_data * len(pd.read_csv(args.data_path + "/snli/test" + s + m + ".txt", sep = "\t", usecols = [0])) / 100)

		# load pre-processed SNLI dataset
		dtype = {"ntokens1": int, "ntokens2": int, "label": int}
		converters = {"sentence1": literal_eval, "sentence2": literal_eval}
		self.SNLI = {
			"train": pd.read_csv(args.data_path + "/snli/train" + s + m + ".txt", sep = "\t", index_col = 0, nrows = nrows_train, dtype = dtype, converters = converters),
			"dev": pd.read_csv(args.data_path + "/snli/dev" + s + m + ".txt", sep = "\t", index_col = 0, nrows = nrows_dev, dtype = dtype, converters = converters),
			"test": pd.read_csv(args.data_path + "/snli/test" + s + m + ".txt", sep = "\t", index_col = 0, nrows = nrows_test, dtype = dtype, converters = converters)
			}

		# dictionary for SNLI dataset sizes
		self.size = {"train": len(self.SNLI["train"]), "dev": len(self.SNLI["dev"]), "test": len(self.SNLI["test"])}

		# load full vocab
		with open(args.data_path + "/vocabulary" + s + ".txt", "r") as f:
			self.n_vocab_full = len(f.read().splitlines())

		# load created embeddings
		self.embeddings = pd.read_csv(args.data_path  + "/embeddings" + s + ".csv", sep = " ", index_col = 0)
		self.embeddings.index.name = "token"
		self.n_vocab, self.n_dim = self.embeddings.shape

		# actual vocab
		self.vocab = set(self.embeddings.index)

		# initialize index for batch scheduler
		self.index = 0

		print("   Finished loading SNLI datatset with train: {}, development: {}, test: {}".format(self.size["train"], self.size["dev"], self.size["test"]))

		print("   Finished loading {}-d embeddings for {} tokens out of {}".format(self.n_dim, self.n_vocab, self.n_vocab_full))

	def get_next_batch(self, data_set, batch_size):

		# get batch from datafame
		batch_SNLI = self.SNLI[data_set][self.index: self.index + batch_size]

		# increase index (for the next batch)
		self.index += batch_size

		# initialize list (tuple) to contain the two batches of sentences
		batch_input, batch_sent_lens = [], []

		# target labels in numpy arrays (not one-hot encoded)
		batch_target = batch_SNLI.label.values

		for sset in [1, 2]:
			batch_i = batch_SNLI[["sentence" + str(sset), "ntokens" + str(sset)]]

			# sentence lengths
			sent_lens = batch_i.loc[:, "ntokens" + str(sset)].values

			# get max sentence_length in the batch
			max_len = int(sent_lens.max())

			# initialize batch embeddings
			batch_emb = np.zeros([max_len, batch_size, self.n_dim])

			# creates a pandas series with length batch size
			# each element in the series is a numpy array of size [sentence_length x n_dim]
			sentences_emb = batch_i["sentence" + str(sset)].apply(lambda x: self.embeddings.reindex(x))

			# put each element of the series in the batch_emb 3d array
			for i in range(batch_size):
				batch_emb[:batch_i["ntokens" + str(sset)].iloc[i], i, :] = sentences_emb.iloc[i]

			batch_input.append(batch_emb)
			batch_sent_lens.append(sent_lens)

		return batch_input, batch_target, batch_sent_lens
		

	def reset_train_set(self):
		self.SNLI["train"] = self.SNLI["train"].iloc[np.random.permutation(len(self.SNLI["train"])), :]
		self.index = 0

#######################################
#######################################

# functions for experiments on the test set
# nothing to do with the training process

	def erase_words(self, n, data_set):

		self.SNLI[data_set] = self.SNLI[data_set].loc[(self.SNLI[data_set][["ntokens1", "ntokens2"]] > 10).all(axis = 1), :]

		def map_unk(x, n):
			sentence = x[0]
			ntokens = x[1]
			for pos in np.random.choice(np.arange(1, ntokens - 1), size = n):
				sentence[pos] = "<unk>"
			return sentence


		self.SNLI[data_set].loc[:, "sentence1"] = self.SNLI[data_set][["sentence1", "ntokens1"]].apply(lambda x: map_unk(x, n), axis = 1)
		self.SNLI[data_set].loc[:, "sentence2"] = self.SNLI[data_set][["sentence2", "ntokens2"]].apply(lambda x: map_unk(x, n), axis = 1)

		self.size[data_set] = len(self.SNLI[data_set])

	def filter_based_on_length(self, data_set, min_len, max_len):

		cond1 = self.SNLI[data_set][["ntokens1", "ntokens2"]].mean(axis = 1) > min_len
		cond2 = self.SNLI[data_set][["ntokens1", "ntokens2"]].mean(axis = 1) < max_len

		self.SNLI[data_set] = self.SNLI[data_set].loc[pd.concat([cond1, cond2], axis = 1).all(axis  =1), :]

		self.size[data_set] = len(self.SNLI[data_set])
