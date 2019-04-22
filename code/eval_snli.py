import numpy as np
import torch
import argparse
from data import Dataset
from models import MLP
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy(logits, targets):
	return torch.mean((torch.max(logits, 1)[1] == targets).double())

def evaluate(dataset, model, batch_size, data_set):

	# reset index
	dataset.index = 0

	# initialize running metrics
	running_accu = 0

	steps = dataset.size[data_set] // batch_size
	for step in range(steps):

		# get batch of sentences and targets
		batch_x, batch_y, batch_lens = dataset.get_next_batch(data_set, batch_size)

		# to tensors
		x1 = torch.tensor(batch_x[0], dtype = torch.float32).to(device)
		x2 = torch.tensor(batch_x[1], dtype = torch.float32).to(device)
		y = torch.tensor(batch_y, dtype = torch.long).to(device)
		len1 = torch.tensor(batch_lens[0], dtype = torch.long).to(device)
		len2 = torch.tensor(batch_lens[1], dtype = torch.long).to(device)

		with torch.no_grad():

			logits = model.forward(x1, x2, len1, len2)

			accu = accuracy(logits, y)

			running_accu += accu.item()

	current_accu = running_accu/steps

	return current_accu

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type = str, default = "./../outputs/biLSTM_maxp/biLSTM_maxp_Adam_X/biLSTM_maxp_bestModel.pwf")
parser.add_argument('--data_set', type = str, default = "test")
parser.add_argument('--encoder', type = str, choices = ["BoW", "LSTM", "backwardLSTM", "biLSTM", "biLSTM_maxp", "biLSTM_minmax"], default = "biLSTM_maxp")
parser.add_argument('--emb_dim', type = int, default = 300)
parser.add_argument('--num_classes', type = int, default = 3)
parser.add_argument('--classifier_hidden_sizes', type = list, default = [512])
parser.add_argument('--classifier_dropout_rate', type = float, default = 0.)
parser.add_argument('--classifier_batch_norm', type = bool, default = False)
parser.add_argument('--classifier_actv_fun', type = str, choices = ["ReLU", "tanh", "linear"], default = "ReLU")
parser.add_argument('--classifier_weight_decay', type = float, default = 0.01)
parser.add_argument('--lstm_hidden_size', type = int, default = 2048)
parser.add_argument('--lstm_num_layers', type = int, default = 1)
parser.add_argument('--lstm_dropout_rate', type = float, default = .0)
parser.add_argument('--lstm_weight_decay', type = float, default = .0001)
parser.add_argument('--grad_check_freq', type = int, default = 2000)
parser.add_argument('--optimizer', type = str, choices = ["Adam", "SGD"], default = "Adam")
parser.add_argument('--max_epochs', type = int, default = 20)
parser.add_argument('--max_epochs_no_incease', type = int, default = 3)
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--learning_rate_decay', type = float, default = 0.99)
parser.add_argument('--lr_reduction_factor', type = float, default = 5.)
parser.add_argument('--lower_lr_limit', type = float, default = 1e-5)
parser.add_argument('--gradient_max_norm', type = float, default = 5.)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--percentage_of_train_data', type = float, default = 0.)
parser.add_argument('--percentage_of_dev_data', type = float, default = 0.)
parser.add_argument('--percentage_of_test_data', type = float, default = 100)
parser.add_argument('--data_path', type = str, default = "./../preprocessed_data")
parser.add_argument('--output_path', type = str, default = "output/")
parser.add_argument('--lowercase', type = bool, default = True)
parser.add_argument('--unknown_mapped', type = bool, default = True)
parser.add_argument('--save_encoder', type = bool, default = True)
args, unparsed = parser.parse_known_args()

emb_dim = 300
lstm_hidden_size = 2048
lstm_num_layers = 1
lstm_dropout_rate = 0
emb_dim = 300
batch_size = args.batch_size
data_set = args.data_set

P = {}

for encoder in ["BoW", "LSTM", "biLSTM", "biLSTM_maxp"]:

	args.encoder = encoder
	P[encoder] = []

	if encoder == "BoW":
		args.classifier_dropout_rate = 0.1
	else:
		args.classifier_dropout_rate = .0

	args.checkpoint_path = "./../outputs/{}/{}_Adam_X/{}_bestModel.pwf".format(encoder, encoder, encoder)

	model = MLP(args)
	model.load_state_dict(torch.load(args.checkpoint_path, map_location = device))
	model.to(device)
	model.eval()

	for n_unk in range(0, 9):

		dataset = Dataset(args)
		if n_unk > 0:
			dataset.erase_words(n_unk, data_set)

		accu = evaluate(dataset, model, args.batch_size, args.data_set)

		P[encoder].append(accu)

		print(encoder, n_unk, accu)


sns.set()
plt.figure(1)
for encoder in ["BoW", "LSTM", "biLSTM", "biLSTM_maxp"]:
	plt.plot(P[encoder], label = encoder)
plt.legend()
plt.xlabel("number of removed tokens")
plt.ylabel("accuracy")
plt.savefig("perfromance.png", dpi = 400)


P = {}

for encoder in ["BoW", "LSTM", "biLSTM", "biLSTM_maxp"]:

	args.encoder = encoder
	P[encoder] = []

	if encoder == "BoW":
		args.classifier_dropout_rate = 0.1
	else:
		args.classifier_dropout_rate = .0

	args.checkpoint_path = "./../outputs/{}/{}_Adam_X/{}_bestModel.pwf".format(encoder, encoder, encoder)

	model = MLP(args)
	model.load_state_dict(torch.load(args.checkpoint_path, map_location = device))
	model.to(device)
	model.eval()

	for min_len, max_len in zip([0, 10, 15, 20], [10, 15, 20, 100]):


		dataset = Dataset(args)

		dataset.filter_based_on_length(data_set, min_len, max_len)

		accu = evaluate(dataset, model, args.batch_size, args.data_set)

		P[encoder].append(accu)

		print(encoder, min_len, max_len, accu)


sns.set()
plt.figure(1)
for encoder in ["BoW", "LSTM", "biLSTM", "biLSTM_maxp"]:
	plt.plot(P[encoder], label = encoder)
plt.legend()
plt.xlabel("mean number of tokens")
plt.ylabel("accuracy")
plt.xticks(np.arange(4), ("0-10", "10-15", "15-20", "20-max"))
plt.savefig("perfromance_N.png", dpi = 400)