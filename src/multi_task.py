import torch
import torch.nn as nn
from torchvision.models import resnet50
from utils import EarlyStopping, get_data_loaders, save_model
import os
from tqdm import tqdm
import numpy as np


class MultiTask(nn.Module):
	def __init__(self):
		super().__init__()
		resnet = resnet50(pretrained=True)

		self.base_model = resnet
		in_feat = resnet.fc.in_features
		self.base_model.fc = nn.Identity()
		self.base_model.head1 = nn.Linear(in_features=in_feat, out_features=40)
		self.base_model.head2 = nn.Linear(in_features=in_feat, out_features=10)

		# self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr=0.01, momentum=0.9)
		self.optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=1e-3)
		self.criterion1 = nn.BCELoss()
		self.criterion2 = nn.MSELoss()
		self.sigm = nn.Sigmoid()
		self.path = os.path.join("../models", "multitask2")
		os.makedirs(self.path, exist_ok=True)

	def forward(self, x):
		x1 = self.sigm(self.base_model.head1(self.base_model(x)))
		x2 = self.base_model.head2(self.base_model(x))
		return x1, x2

	def get_metric(self, all_preds, all_labels):
		pass


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	train_dataloader, eval_dataloader = get_data_loaders(root='data', download=False,
	                                                     target_type=["attr", "landmarks"], batch_size=32, shuffle=True,
	                                                     num_workers=8)
	model = MultiTask()

	patience = 5
	early_stopper = EarlyStopping(patience=patience, verbose=True)
	num_epochs = 50

	model.to(device)
	metrics = {
		"train_loss": [],
		"eval_loss": [],
	}
	loss_factors = [0.2, 0.8]
	for epoch in range(num_epochs):
		train_epoch_loss, eval_epoch_loss = [], []

		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		model.train()
		for inputs, labels in tqdm(train_dataloader):
			attr_labels, lm_labels = labels

			inputs = inputs.to(device)
			lm_labels = lm_labels.to(device)
			attr_labels = attr_labels.to(device)

			model.optimizer.zero_grad()

			attr_output, lm_output = model(inputs)

			attr_loss = model.criterion1(attr_output, attr_labels.type(torch.float))
			lm_loss = model.criterion2(lm_output, lm_labels.type(torch.float))
			if lm_loss < 1.0:
				loss_factors = [0.5, 0.5]

			loss = loss_factors[0] * lm_loss + loss_factors[1] * attr_loss
			train_epoch_loss.append(loss.item())

			loss.backward()
			model.optimizer.step()
		metrics["train_loss"].append(np.mean(train_epoch_loss))
		print('{} Loss: {:.4f}'.format("Train", metrics["train_loss"][epoch]))

		model.eval()
		with torch.no_grad():
			for inputs, labels in tqdm(eval_dataloader):
				attr_labels, lm_labels = labels

				inputs = inputs.to(device)
				lm_labels = lm_labels.to(device)
				attr_labels = attr_labels.to(device)
				attr_output, lm_output = model(inputs)
				attr_loss = model.criterion1(attr_output, attr_labels.type(torch.float))
				lm_loss = model.criterion2(lm_output, lm_labels.type(torch.float))

				loss = loss_factors[0] * lm_loss + loss_factors[1] * attr_loss
				eval_epoch_loss.append(loss.item())
		metrics["eval_loss"].append(np.mean(eval_epoch_loss))

		early_stopper(np.mean(eval_epoch_loss), model, path=os.path.join(model.path, f"epoch-{epoch}.pt"))
		print('{} Loss: {:.4f}'.format("Train", metrics["train_loss"][epoch]))
		print('{} Loss: {:.4f}'.format("Eval", metrics["eval_loss"][epoch]))

		if early_stopper.early_stop:
			print(f"Early stopping: Epoch {epoch - 1}")
			break
	save_model(metrics, model, plots=False)


if __name__ == "__main__":
	main()
