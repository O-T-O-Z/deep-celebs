import torch
from utils import get_data_loaders
from tqdm import tqdm
from PIL import ImageDraw
import numpy as np
from landmarks import LandmarkPredictor
from multi_task import MultiTask
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


def plot_landmarks(input, output, label):
	output = output.detach().cpu().numpy()
	label = label.detach().cpu().numpy()

	draw = ImageDraw.Draw(input)
	for coord in output:
		draw.ellipse((coord[0] - 3, coord[1] - 3, coord[0] + 3, coord[1] + 3), fill='red')

	input.save("pred.jpg")
	for coord in label:
		draw.ellipse((coord[0] - 3, coord[1] - 3, coord[0] + 3, coord[1] + 3), fill='green')
	input.save("label.jpg")


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	test_data = CelebA(root='data', split="test", download=False, transform=transform, target_type=["landmarks"])
	test_data_no_transform = CelebA(root='data', split="test", download=False, target_type=["landmarks"])
	# model = LandmarkPredictor(n_classes=10)
	model = MultiTask()
	model.load_state_dict(torch.load('models/multitask/best_model.pt', map_location=device))

	model.eval()
	model.to(device)
	test_img_idx = 0
	with torch.no_grad():
		input, label = test_data[test_img_idx]
		input = input[None, :]
		input = input.to(device)
		label = label.to(device)
		output = model(input)
		if model.__class__.__name__ == "MultiTask":
			output = output[1]
		output = output.reshape(-1, 2)
		label = label.reshape(-1, 2)
		if model.__class__.__name__ == "MultiTask":
			loss = model.criterion2(output, label)
		else:
			loss = model.criterion(output, label)
		plot_landmarks(test_data_no_transform[test_img_idx][0], output, label)
		print(f"MSE loss: {loss.item()}")


if __name__ == "__main__":
	main()
