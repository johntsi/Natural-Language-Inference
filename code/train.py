import argparse
import numpy as np
from os import makedirs
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import clock
from datetime import datetime
from copy import copy

from data import Dataset
from models import MLP

def plot_grad_flow(named_parameters, step):
	'''Plots the gradients flowing through different layers in the net during training.
	Can be used for checking for possible gradient vanishing / exploding problems.
	Taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10'''
	ave_grads = []
	max_grads= []
	layers = []
	for n, p in named_parameters:
		if(p.requires_grad) and ("bias" not in n):
			layers.append(n)
			ave_grads.append(p.grad.abs().mean())
			max_grads.append(p.grad.abs().max())
	plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
	plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlim(left=0, right=len(ave_grads))
	plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	plt.legend([Line2D([0], [0], color="c", lw=4),
				Line2D([0], [0], color="b", lw=4),
				Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

	# output/Encoder/Encoder_hyperparameters/gradient/epochX_stepY.png
	plt.savefig(save_path + "gradient/epoch" + str(epoch) + "_step" + str(step) + ".png", dpi = 400)

def accuracy(logits, targets):
	return torch.mean((torch.max(logits, 1)[1] == targets).double())

def training_epoch(dataset, model, optimizer, loss_fn, args):

	# shuffle and reset index of train set
	dataset.reset_train_set()

	# initialize running metrics
	running_loss, running_accu = 0, 0

	# enter training mode
	model = model.train()

	steps = dataset.size["train"] // args.batch_size
	for step in range(steps):

		# get batch of sentences and targets
		batch_x, batch_y, batch_lens = dataset.get_next_batch("train", args.batch_size)

		# to tensors
		x1 = torch.tensor(batch_x[0], dtype = torch.float32).to(device)
		x2 = torch.tensor(batch_x[1], dtype = torch.float32).to(device)
		y = torch.tensor(batch_y, dtype = torch.long).to(device)
		len1 = torch.tensor(batch_lens[0], dtype = torch.long).to(device)
		len2 = torch.tensor(batch_lens[1], dtype = torch.long).to(device)

		logits = model.forward(x1, x2, len1, len2)

		loss = loss_fn(logits, y)
		accu = accuracy(logits, y)

		running_loss += loss.item()
		running_accu += accu.item()

		optimizer.zero_grad()
		loss.backward()

		# Modified SGD
		# does clipping and normalizes gradients
		if args.optimizer == "SGD":

			shrink_factor = 1
			total_norm = 0

			for param in model.parameters():
				if param.requires_grad:
					param.grad.data.div_(args.batch_size)
					total_norm += param.grad.data.norm()**2
			total_norm = np.sqrt(total_norm)

			if total_norm > args.gradient_max_norm:
				shrink_factor = args.gradient_max_norm / total_norm
			current_learning_rate = optimizer.param_groups[0]["lr"]
			optimizer.param_groups[0]["lr"] = current_learning_rate * shrink_factor

		# plot gradient flow every X steps
		if not (step + 1) % args.grad_check_freq:
			plot_grad_flow(model.named_parameters(), step)

		optimizer.step()

		# recover learning rate
		if args.optimizer == "SGD":
			optimizer.param_groups[0]["lr"] = current_learning_rate

	return running_loss/steps, running_accu/steps


def evaluation_step(dataset, model, data_set, loss_fn, args):

	# reset index
	dataset.index = 0

	# enter evaluation mode
	model = model.eval()

	# initialize running metrics
	running_loss, running_accu = 0, 0

	steps = dataset.size[data_set] // args.batch_size
	for step in range(steps):

		# get batch of sentences and targets
		batch_x, batch_y, batch_lens = dataset.get_next_batch(data_set, args.batch_size)

		# to tensors
		x1 = torch.tensor(batch_x[0], dtype = torch.float32).to(device)
		x2 = torch.tensor(batch_x[1], dtype = torch.float32).to(device)
		y = torch.tensor(batch_y, dtype = torch.long).to(device)
		len1 = torch.tensor(batch_lens[0], dtype = torch.long).to(device)
		len2 = torch.tensor(batch_lens[1], dtype = torch.long).to(device)

		with torch.no_grad():

			logits = model.forward(x1, x2, len1, len2)

			loss = loss_fn(logits, y)
			accu = accuracy(logits, y)

			running_loss += loss.item()
			running_accu += accu.item()

	current_loss = running_loss/steps
	current_accu = running_accu/steps

	# save if this is the best model so far (start saving after tha mid-training point for efficiency)
	if data_set == "dev":
		if current_accu > best_dev_accu:
			# output/Encoder_hyperparameters/bestmodel.pwf
			torch.save(model.state_dict(), save_path + args.encoder + "_bestModel.pwf")

	return current_loss, current_accu


def train(args):

	global best_dev_accu, save_path, epoch, device

	# output/Encoder/Encoder_hyperparameters/
	save_path = args.output_path + "/" + args.encoder + "/" + args.encoder + "_" + str(args.optimizer) + "_" + "X" + "/"
	makedirs(save_path)

	# directory for gradients
	# output/Encoder/Encoder_hyperparameters/gradient
	makedirs(save_path + "gradient")

	best_dev_accu = .0

	t = clock()

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("Device: {}".format(device))

	print("Initializing dataset ...")
	dataset = Dataset(args)

	print("Initializing model ...")
	model = MLP(args).to(device)

	print("Model's state_dict:")
	for name, param in model.named_parameters():
		print("   ", name, param.requires_grad, param.data.shape)
	print("_"*50)

	# Dictionary to store metrics during training and evaluation
	Metrics = {"Accuracy": {"train": [], "dev": []}, "Loss": {"train": [], "dev": []} }

	# standard Cross entropy loss function
	loss_fn = torch.nn.CrossEntropyLoss()

	current_learning_rate = args.learning_rate

	# Setup optimizer
	# different weight decay for classification and encoder
	if args.optimizer == "SGD":
		if args.encoder == "BoW":
			optimizer = torch.optim.SGD(model.parameters(), lr = current_learning_rate, weight_decay = args.classifier_weight_decay)
		else:
			optimizer = torch.optim.SGD([{'params': model.layers.parameters()}, 
										 {"params": model.encoder.parameters(), "weight_decay": args.lstm_weight_decay}],
										 lr = current_learning_rate, weight_decay = args.classifier_weight_decay)
	else:
		if args.encoder == "BoW":
			optimizer = torch.optim.SGD(model.parameters(), lr = current_learning_rate, weight_decay = args.classifier_weight_decay)
		else:
			optimizer = torch.optim.Adam([{'params': model.layers.parameters()}, 
										 {"params": model.encoder.parameters(), "weight_decay": args.lstm_weight_decay}],
										 lr = current_learning_rate, weight_decay = args.classifier_weight_decay)

	epoch = -1
	stop_training = False
	epochs_no_increase = 0

	# different stopping conditons depending on the optimizer
	# lower learning rate for SGD
	# max epochs with no increase in dev accuracy for Adam
	while ((current_learning_rate > args.lower_lr_limit) and args.optimizer == "SGD") or (not stop_training and (args.optimizer == "Adam")):

		epoch += 1

		# one epoch in trains set, running mean of loss and accu
		loss_train, accu_train = training_epoch(dataset, model, optimizer, loss_fn, args)

		# evaluation on development and test sets
		loss_dev, accu_dev = evaluation_step(dataset, model, "dev", loss_fn, args)

		# store metrics
		Metrics["Accuracy"]["train"].append(accu_train)
		Metrics["Accuracy"]["dev"].append(accu_dev)
		Metrics["Loss"]["train"].append(loss_train)
		Metrics["Loss"]["dev"].append(loss_dev)

		# Adam
		if (epoch > 0) and (args.optimizer == "Adam"):
			if Metrics["Accuracy"]["dev"][-1] < best_dev_accu:
				epochs_no_increase += 1
			else:
				epochs_no_increase = 0
				best_dev_accu = copy(Metrics["Accuracy"]["dev"][-1])

			if epochs_no_increase == args.max_epochs_no_incease:
				stop_training = True

		if epoch == args.max_epochs:
			stop_training = True


		# SGD
		# reduce a lot learning rate if dev accuracy dropped
		# reduce just a bit if not
		if (epoch > 0) and (args.optimizer == "SGD"):
			if Metrics["Accuracy"]["dev"][-1] < Metrics["Accuracy"]["dev"][-2]:
				current_learning_rate /= args.lr_reduction_factor
				print("     --> learning rate reduced to {:.6f} due to drop in dev set accuracy.".format(current_learning_rate))
			else:
				current_learning_rate *= args.learning_rate_decay
			for param_group in optimizer.param_groups:
				param_group["lr"] = current_learning_rate

		print("[{}] epoch {} || LOSS: train = {:.4f}, dev = {:.4f} || ACCURACY: train = {:.4f}, dev = {:.4f}".format(
			datetime.now().time().replace(microsecond = 0), epoch, loss_train, loss_dev, accu_train, accu_dev))

	# re-init model
	model = MLP(args).to(device)

	# load best model
	# output/Encoder/Encoder_hyperparameters/bestmodel.pwf
	model.load_state_dict(torch.load(save_path + args.encoder + "_bestModel.pwf"))

	# TEST set perfromance
	loss_test, accu_test = evaluation_step(dataset, model, "test", loss_fn, args)

	print("Training completed in {:.2f} minutes.".format((clock() - t)/60))
	print("Test set performance: Loss = {:.4f} || Accuracy = {:.4f}".format(loss_test, accu_test))

	if args.save_encoder:
		# output/Encoder/Encoder_hyperparameters/bestEncoder.pwf
		torch.save(model.encoder.state_dict(), save_path + args.encoder + "_bestEncoder.pwf")

	# plot training curves
	sns.set()
	fig, ax = plt.subplots(1, 2)
	for i, metric, metric_test in zip([0, 1], ["Loss", "Accuracy"], [loss_test, accu_test]):
		for data_set in Metrics[metric].keys():
			ax[i].plot(Metrics[metric][data_set], label = data_set)
		ax[i].scatter(len(Metrics[metric][data_set]), metric_test, s = 50, marker = "x", label = "test", color = "k")
		ax[i].set_title(metric)
		ax[i].legend()
	bottom = np.min(Metrics["Loss"]["train"]) * 0.8
	top = np.max([np.median(Metrics["Loss"]["train"]), np.median(Metrics["Loss"]["dev"])]) * 2
	ax[0].set_ylim(bottom = bottom, top = top)

	# output/Encoder/Encoder_hyperparameters/training.png
	fig.savefig(save_path + "training.png", dpi = 400)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--encoder', type = str, choices = ["BoW", "LSTM", "biLSTM", "biLSTM_maxp"], default = "biLSTM_maxp")
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
	parser.add_argument('--batch_size', type = int, default = 64)
	parser.add_argument('--percentage_of_train_data', type = float, default = 100.)
	parser.add_argument('--percentage_of_dev_data', type = float, default = 100.)
	parser.add_argument('--percentage_of_test_data', type = float, default = 100.)
	parser.add_argument('--data_path', type = str, default = "./../preprocessed_data")
	parser.add_argument('--output_path', type = str, default = "output/")
	parser.add_argument('--lowercase', type = bool, default = True)
	parser.add_argument('--unknown_mapped', type = bool, default = True)
	parser.add_argument('--save_encoder', type = bool, default = True)
	args, unparsed = parser.parse_known_args()

	print("*** Hyperparameters ***")
	print("_"*50)
	for key, value in vars(args).items():
		print(key + ' : ' + str(value))
	print("_"*50)

	train(args)