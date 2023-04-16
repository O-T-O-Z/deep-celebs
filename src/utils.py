import numpy as np
import torch
from tqdm import tqdm
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import matplotlib.pyplot as plt


# The class below has been adapted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
	"""
	Early stops the training process while also saving the model to a specified checkpoint.
	"""

	def __init__(self, patience: int, verbose: bool, delta: int = 0):
		self.patience = patience
		self.verbose = verbose
		self.delta = delta
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf

	def __call__(self, val_loss: float, model, path: str = "checkpoint.pt"):
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, path)

		elif score < self.best_score + self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, path)
			self.counter = 0

	def save_checkpoint(self, val_loss: float, model, path: str):
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), path)
		self.val_loss_min = val_loss


def get_data_loaders(root, download, target_type, batch_size, shuffle, num_workers):
	"""
	Returns dataloaders for the train and eval sets.

	:param root: root of the dataset.
	:param download: whether to download the dataset.
	:param target_type: type of target to use, i.e. "attr", "landmarks".
	:param batch_size: batch size.
	:param shuffle: shuffle the data.
	:param num_workers: number of workers.
	:return: train and eval dataloaders.
	"""

	# we follow the procedure as presented by https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	train_data = CelebA(root=root, split="train", download=download, transform=transform, target_type=target_type)
	train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	eval_data = CelebA(root=root, split="valid", download=download, transform=transform, target_type=target_type)
	eval_data_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	return train_data_loader, eval_data_loader


def train_model(model, data_loader, valid_data_loader, early_stopper, num_epochs, device):
	"""
	Executes the training loop.

	:param model: model to train.
	:param data_loader: data loader for the train set.
	:param valid_data_loader: data loader for the eval set.
	:param early_stopper: early stopping object.
	:param num_epochs: number of epochs.
	:param device: device to use.
	:return: metrics of the training.
	"""
	model.to(device)
	metrics = {
		"train_loss": [],
		"train_metric": [],
		"eval_loss": [],
		"eval_metric": []
	}
	for epoch in range(num_epochs):
		train_epoch_loss, eval_epoch_loss = [], []
		all_preds, all_labels = [], []

		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		model.train()
		for inputs, labels in tqdm(data_loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			model.optimizer.zero_grad()
			outputs = model(inputs)

			all_preds.append(torch.round(outputs))
			all_labels.append(labels)

			loss = model.criterion(outputs, labels.type(torch.float))
			train_epoch_loss.append(loss.item())

			loss.backward()
			model.optimizer.step()
		metrics["train_loss"].append(np.mean(train_epoch_loss))
		metrics["train_metric"].append(model.get_metric(all_preds, all_labels))
		all_preds, all_labels = [], []

		model.eval()
		with torch.no_grad():
			for inputs, labels in tqdm(valid_data_loader):
				inputs = inputs.to(device)
				labels = labels.to(device)
				outputs = model(inputs)

				all_preds.append(torch.round(outputs))
				all_labels.append(labels)

				loss = model.criterion(outputs, labels.type(torch.float))
				eval_epoch_loss.append(loss.item())
		metrics["eval_loss"].append(np.mean(eval_epoch_loss))
		metrics["eval_metric"].append(model.get_metric(all_preds, all_labels))

		early_stopper(np.mean(eval_epoch_loss), model, path=os.path.join(model.path, f"epoch-{epoch}.pt"))
		print(
			'{} Metric {} - Loss: {:.4f}'.format("Train", metrics["train_metric"][epoch], metrics["train_loss"][epoch]))
		print('{} Metric {} - Loss: {:.4f}'.format("Eval", metrics["eval_metric"][epoch], metrics["eval_loss"][epoch]))

		if early_stopper.early_stop:
			print(f"Early stopping: Epoch {epoch - 1}")
			break
	return metrics


def save_model(metrics, model, plots=True):
	"""
	Saves the model and the metrics.

	:param metrics: metrics of the training.
	:param model: model to save.
	:param plots: whether to save the plots.
	:return:
	"""
	with open(os.path.join(model.path, "metrics.json"), "w") as f:
		json.dump(metrics, f)
	if plots:
		plot_metrics(metrics, model.path)


def plot_metrics(metrics, path):
	"""
	Plots the metrics.

	:param metrics: metrics of the training.
	:param path: path to save the plots.
	"""
	fig, ax = plt.subplots(1, 2, figsize=(15, 5))
	ax[0].plot(metrics["train_loss"], label="Train")
	ax[0].plot(metrics["eval_loss"], label="Eval")
	ax[0].set_title("Loss")
	ax[0].set_xlabel("Epoch")
	ax[0].legend()
	ax[1].plot(metrics["train_metric"], label="Train")
	ax[1].plot(metrics["eval_metric"], label="Eval")
	ax[1].set_title("Metric")
	ax[1].set_xlabel("Epoch")
	ax[1].legend()
	plt.savefig(os.path.join(path, "metrics.png"))
	plt.close()


def calculate_f1(attribute_metrics):
	"""
	Calculates the F1 score, precision, and recall.

	:param attribute_metrics: metrics of the attributes.
	:return: metrics of the attributes with F1, precision, and recall.
	"""
	for key in attribute_metrics.keys():
		attr = attribute_metrics[key]
		if attr["TP"] != 0:
			attr["precision"] = attr["TP"] / (attr["TP"] + attr["FP"])
			attr["recall"] = attr["TP"] / (attr["TP"] + attr["FN"])
			attr["f1"] = (2 * attr["precision"] * attr["recall"]) / (attr["precision"] + attr["recall"])
		else:
			attr["precision"] = 0
			attr["recall"] = 0
			attr["f1"] = 0
	return attribute_metrics


def calculate_cm(pred, label, attribute_metrics):
	"""
	Calculates the confusion matrix values.

	:param pred: predictions.
	:param label: labels.
	:param attribute_metrics: metrics of the attributes.
	:return: metrics of the attributes with the confusion matrix values.
	"""
	attributes_list = list(attribute_metrics.keys())
	for j_i, j in enumerate(pred):
		for k_i, k in enumerate(j):
			p = k.item()
			l = label[j_i][k_i].item()
			if p == 1 and l == 1:
				attribute_metrics[attributes_list[k_i]]["TP"] += 1
			elif p == 1 and l == 0:
				attribute_metrics[attributes_list[k_i]]["FP"] += 1
			elif p == 0 and l == 1:
				attribute_metrics[attributes_list[k_i]]["FN"] += 1
	return attribute_metrics
