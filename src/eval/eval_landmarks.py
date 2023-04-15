import torch
from tqdm import tqdm
import numpy as np
from landmarks import LandmarkPredictor
from multi_task import MultiTask
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	test_data = CelebA(root='data', split="test", download=False, transform=transform, target_type=["landmarks"])
	test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8)
	model = LandmarkPredictor(n_classes=10)
	# model = MultiTask()
	model.load_state_dict(torch.load('../../models/landmarksSGD/best_model.pt', map_location=device))

	model.eval()
	model.to(device)
	with torch.no_grad():
		losses = []
		for inputs, labels in tqdm(test_data_loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)

			outputs = outputs.reshape(-1, 2)
			labels = labels.reshape(-1, 2)
			loss = model.criterion(outputs, labels)
			losses.append(loss.item())
	average_loss = np.mean(losses)
	print(f"Average MSE loss: {average_loss}")


if __name__ == "__main__":
	main()
