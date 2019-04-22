import pandas as pd 
import numpy as np
from ast import literal_eval

data_path = "./../preprocessed_data"
s = "_lower"

# load SNLI preprocessed
dtype = {"ntokens1": int, "ntokens2": int, "label": int}
converters = {"sentence1": literal_eval, "sentence2": literal_eval}
SNLI = {
	"train": pd.read_csv(data_path + "/snli/train" + s + ".txt", sep = "\t", index_col = 0, dtype = dtype, converters = converters),
	"dev": pd.read_csv(data_path + "/snli/dev" + s + ".txt", sep = "\t", index_col = 0, dtype = dtype, converters = converters),
	"test": pd.read_csv(data_path + "/snli/test" + s + ".txt", sep = "\t", index_col = 0, dtype = dtype, converters = converters)
	}

# load embeddings preprocessed
embeddings = pd.read_csv(data_path  + "/embeddings" + s + ".csv", sep = " ", index_col = 0, usecols = [0, 1])

# function to map to each unkown token in a sentence, the token <unk>
def mapf(x):
	y = np.array(x[0])
	y[x[1]] = "<unk>"
	return y.tolist()

for data_set in SNLI.keys():
	for sset in [1, 2]:

		# create column that indicates the position of unkown tokens in each sentence
		SNLI[data_set]["presence" + str(sset)] = SNLI[data_set].loc[:, "sentence" + str(sset)].apply(
			lambda x: pd.isnull(embeddings.iloc[:, 0].reindex(x)).values)

		# map unkown token to these positions
		SNLI[data_set].loc[:, "sentence" + str(sset)] = SNLI[data_set].loc[:, ["sentence" + str(sset), "presence" + str(sset)]].apply(mapf, axis = 1)

		# delete helper column
		del SNLI[data_set]["presence" + str(sset)]

	# correct 4 inconsistencies in train set (<unk instead of <unk>)
	if data_set == "train":
		for idx, pos in zip([124462, 128180, 393745, 480568], [2, 6, 4, 2]):
			SNLI[data_set].loc[idx, "sentence2"][pos] = "<unk>"

	SNLI[data_set].to_csv(data_path + "/snli/" + data_set + s + "_mapped" + ".txt", sep = "\t")