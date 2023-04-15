import torch
import torch.nn as nn
from torchvision.models import resnet50
from utils import EarlyStopping, train_model, get_data_loaders, save_model
import os


class AttributeClassifier(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		resnet = resnet50(pretrained=True)
		resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
		self.base_model = resnet
		self.sigm = nn.Sigmoid()
		self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr=0.01, momentum=0.9)
		# self.optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=1e-3)
		self.criterion = nn.BCELoss()
		self.path = os.path.join("../models", "attributes")
		os.makedirs(self.path, exist_ok=True)

	def forward(self, x):
		return self.sigm(self.base_model(x))

	def get_metric(self, all_preds, all_labels):
		TP, FP, FN = 0, 0, 0
		for i, pred in enumerate(all_preds):
			TP += sum(((pred == 1) & (all_labels[i] == 1)).sum(dim=0)).cpu().item()
			FP += sum(((pred == 1) & (all_labels[i] == 0)).sum(dim=0)).cpu().item()
			FN += sum(((pred == 0) & (all_labels[i] == 1)).sum(dim=0)).cpu().item()

		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		return [precision, recall]


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	train_dataloader, eval_dataloader = get_data_loaders(root='data', download=False,
	                                                     target_type=["attr"], batch_size=32, shuffle=True,
	                                                     num_workers=8)
	model = AttributeClassifier(n_classes=40)

	patience = 5
	early_stopper = EarlyStopping(patience=patience, verbose=True)
	num_epochs = 50

	metrics = train_model(model, train_dataloader, eval_dataloader, early_stopper, num_epochs, device)
	save_model(metrics, model)


if __name__ == "__main__":
	main()
