import pandas as pd 
import numpy as np
import nltk
import os
from ast import literal_eval

# nltk.download('punkt')


def get_snli(lowercase, snli_folder_path):

	# load original SNLI dataset
	SNLI = {"train": pd.read_csv(snli_folder_path + "/snli_1.0_train.txt", sep = "\t"),
			"dev": pd.read_csv(snli_folder_path + "/snli_1.0_dev.txt", sep = "\t"),
			"test": pd.read_csv(snli_folder_path + "/snli_1.0_test.txt", sep = "\t")}

	for data_set in SNLI.keys():

		# keep only pairs with gold labels and only relevant columns
		SNLI[data_set] = SNLI[data_set][SNLI[data_set].gold_label != "-"][["gold_label", "sentence1", "sentence2"]]

		# remove some examples with non-string values at sentence fields
		valid_indices = SNLI[data_set][["sentence1", "sentence2"]].applymap(type).eq(str).prod(axis = 1).astype(bool)
		SNLI[data_set] = SNLI[data_set].loc[valid_indices, :]
		SNLI[data_set].reset_index(inplace = True, drop = True)

		# map column labels to integers
		# entailment: 0, contradiction: 1, neutral: 2
		SNLI[data_set]["label"] = 0
		SNLI[data_set].loc[SNLI[data_set].gold_label == "contradiction", "label"] = 1
		SNLI[data_set].loc[SNLI[data_set].gold_label == "neutral", "label"] = 2

		del SNLI[data_set]["gold_label"]

		for sset in [1, 2]:

			# lowercase if true
			if lowercase:
				SNLI[data_set].loc[:, "sentence" + str(sset)] = SNLI[data_set]["sentence" + str(sset)].str.lower()

			# tokenize sentences
			SNLI[data_set].loc[:, "sentence" + str(sset)] = SNLI[data_set]["sentence" + str(sset)].apply(nltk.word_tokenize)

			# add start and end of sentence indicators
			SNLI[data_set].loc[:, "sentence" + str(sset)] = SNLI[data_set]["sentence" + str(sset)].apply(lambda x: ["<s>"] + x + ["</s>"])

			# add column for the number of tokens (usefull for batching)
			SNLI[data_set]["ntokens" + str(sset)] = SNLI[data_set]["sentence" + str(sset)].apply(len)

	return SNLI


def build_vocabulary(SNLI):

	# create vocabulary
	vocab = set()
	for data_set in SNLI.keys():

		for idx, row in SNLI[data_set].iterrows():

			vocab.update(row["sentence1"])
			vocab.update(row["sentence2"])

	# sort values
	vocab = set(sorted(vocab))

	return vocab


def build_embeddings(vocab, glove_size, glove_path):

	chunksize = 10**4

	for i, chunk in enumerate(pd.read_csv(glove_path, header = None, sep = "\s", index_col = 0, engine = "python",
	 error_bad_lines = False, warn_bad_lines = False, chunksize = chunksize, nrows = glove_size)):

		# get embeddings for the words in vocabulary for this chunk
		if not i:
			embeddings = chunk[chunk.index.isin(vocab)]
		else:
			embeddings = pd.concat([embeddings, chunk[chunk.index.isin(vocab)]], axis = 0)

	# add unknown tokken
	unknown_token = pd.DataFrame(np.random.uniform(-0.05, 0.05, [1, embeddings.shape[1]]), index = ["<unk>"], columns = embeddings.columns)
	embeddings = pd.concat([embeddings, unknown_token], axis = 0)

	# sort index
	embeddings.sort_index(inplace = True)

	return embeddings

####################################################
####################################################

# PROVIDE DATA PATHS
snli_folder_path  = "./../snli_1.0"
glove_path = "C:/Users/ioann/Datasets/glove.840B.300d.txt"


pre_processed_path = "./../preprocessed_data"
lowercase = True
glove_size = None

s = ""
if lowercase:
	s = "_lower"

# SNLI dataframes
SNLI = get_snli(lowercase, snli_folder_path)
for data_set in SNLI.keys():
	SNLI[data_set].to_csv(pre_processed_path + "/snli/" + data_set + s + ".txt", sep = "\t")
print("Finished SNLI. Sizes = {}, {}, {}".format(len(SNLI["train"]), len(SNLI["dev"]), len(SNLI["test"])))

# vocabulary set
vocab = build_vocabulary(SNLI)
with open(pre_processed_path  + "/vocabulary" + s + ".txt", "w") as f:
	for token in vocab:
		f.write(token + "\n")
print("Finished vocabulary. Size = {}".format(len(vocab)))

# sorted embeddings dataframe
embeddings = build_embeddings(vocab, glove_size, glove_path)
embeddings.to_csv(pre_processed_path + "/embeddings" + s + "full.csv", sep = " ")
print("Finished embeddings. Size = {}".format(len(embeddings)))