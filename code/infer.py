import torch
import argparse
import nltk
import numpy as np
import pandas as pd
# nltk.download('punkt')

from models import MLP

def generate_sent_embeddings(sent, word_embeddings, vocab, args):
	emb = np.zeros([len(sent), 1, args.emb_dim])
	known = 0
	for i, token in enumerate(sent):
		if token in vocab:
			emb[i, 0, :] = word_embeddings.loc[token, :]
			known += 1
	return emb, known

def softmax(x):
	exp_x = np.exp(x - np.max(x))
	sum_exp_x = exp_x.sum()
	return exp_x/sum_exp_x

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type = str, default = "./../outputs/biLSTM_maxp/biLSTM_maxp_Adam_X/biLSTM_maxp_bestModel.pwf")
parser.add_argument('--encoder', type = str, choices = ["BoW", "LSTM", "biLSTM", "biLSTM_maxp"], default = "biLSTM_maxp")
parser.add_argument('--emb_dim', type = int, default = 300)
parser.add_argument('--num_classes', type = int, default = 3)
parser.add_argument('--classifier_hidden_sizes', type = list, default = [512])
parser.add_argument('--classifier_dropout_rate', type = float, default = 0.)
parser.add_argument('--classifier_batch_norm', type = bool, default = False)
parser.add_argument('--classifier_actv_fun', type = str, choices = ["ReLU", "tanh", "linear"], default = "ReLU")
parser.add_argument('--lstm_hidden_size', type = int, default = 2048)
parser.add_argument('--lstm_num_layers', type = int, default = 1)
parser.add_argument('--lstm_dropout_rate', type = float, default = .0)
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--data_path', type = str, default = "./../preprocessed_data")
args, unparsed = parser.parse_known_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.checkpoint_path = "./../checkpoints/{}_Adam_X/{}_bestModel.pwf".format(args.encoder, args.encoder)

if args.encoder == "BoW":
	args.classifier_dropout_rate = .1

model = MLP(args)
model.load_state_dict(torch.load(args.checkpoint_path, map_location = device))
model.to(device)
model.eval()

print("Beginning. interactive. session. God-level language understading")
print("*Beep* *beep* *boop* (robot sounds)")
print("_"*50)

print("{} encoder with MLP classifier loaded.".format(args.encoder))

# loading only the embeddings for the words of the SNLI corpus
word_embeddings = pd.read_csv(args.data_path  + "/embeddings_lower.csv", sep = " ", index_col = 0)
vocab = set(word_embeddings.index)

print("{} GloVe word embeddings loaded".format(len(word_embeddings)))

while True:

	print("Please provide a premise and a hypothesis sentence ...")
	premise = input("premise: ").lower()
	hypothesis = input("hypothesis: ").lower()

	premise_tokens = ["<s>"] + nltk.word_tokenize(premise) + ["</s>"]
	hypothesis_tokens = ["<s>"] + nltk.word_tokenize(hypothesis) + ["</s>"]

	premise_length = len(premise_tokens)
	hypothesis_length = len(hypothesis_tokens)

	premise_sent_emb, premise_known = generate_sent_embeddings(premise_tokens, word_embeddings, vocab, args)
	hypothesis_sent_emb, hypothesis_known = generate_sent_embeddings(hypothesis_tokens, word_embeddings, vocab, args)

	print("{}/{} embeddings found for premise.".format(premise_known - 2, premise_length - 2))
	print("{}/{} embeddings found for hypothesis.".format(hypothesis_known - 2, hypothesis_length - 2))

	s1 = torch.tensor(premise_sent_emb, dtype = torch.float32).to(device)
	s2 = torch.tensor(hypothesis_sent_emb, dtype = torch.float32).to(device)
	l1 = torch.tensor([premise_length], dtype = torch.long).to(device)
	l2 = torch.tensor([hypothesis_length], dtype = torch.long).to(device)

	with torch.no_grad():

		logits = model.forward(s1, s2, l1, l2)
		prediction = int(logits.argmax(1).cpu().numpy()[0])
		probability = softmax(logits.cpu().numpy()).squeeze()

	result = {0: "The hypothesis ENTAILS the premise with {:.2f}% probability.".format(probability[prediction] * 100),
	          1: "The hypothesis CONTRADICTS the premise with {:.2f}% probability.".format(probability[prediction] * 100),
	          2: "The relationship of the premise and the hypothesis is NEUTRAL with {:.2f}% probability.".format(probability[prediction] * 100)}[prediction]

	print(result)

	cont = input("Continue? (y/n): ")
	if cont == "n":
		print("End of interactive session.")
		break












