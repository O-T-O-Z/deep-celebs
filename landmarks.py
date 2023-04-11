import torch
import torch.nn as nn
from torchvision.models import resnet50
from utils import EarlyStopping, train_model, get_data_loaders, save_model
import os


class LandmarkPredictor(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		resnet = resnet50(pretrained=True)
		resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
		self.base_model = resnet
		self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr=0.01, momentum=0.9)
		# self.optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=1e-3)
		self.criterion = nn.MSELoss()
		self.path = os.path.join("models", "landmarks")
		os.makedirs(self.path, exist_ok=True)

	def forward(self, x):
		return self.base_model(x)

	def get_metric(self, all_preds, all_labels):
		losses = []
		for i, pred in enumerate(all_preds):
			losses.append(self.criterion(pred, all_labels[i]))
		result = sum(losses) / len(losses)
		# del losses
		return result.cpu().item()


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	train_dataloader, eval_dataloader = get_data_loaders(root='data', download=False,
	                                                     target_type=["landmarks"], batch_size=32, shuffle=True,
	                                                     num_workers=8)
	model = LandmarkPredictor(n_classes=10)

	patience = 5
	early_stopper = EarlyStopping(patience=patience, verbose=True)
	num_epochs = 50

	metrics = train_model(model, train_dataloader, eval_dataloader, early_stopper, num_epochs, device)
	save_model(metrics, model)


if __name__ == "__main__":
	main()
