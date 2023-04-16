import torch
from PIL import ImageDraw
from landmarks import LandmarkPredictor
from multi_task import MultiTask
from torchvision import transforms
from torchvision.datasets import CelebA
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_landmarks(input, output, label):
	output = output.detach().cpu().numpy()
	label = label.detach().cpu().numpy()

	plt.imshow(input)

	for coord in label:
		rect = patches.Ellipse((coord[0], coord[1]), 10, 10, linewidth=2, facecolor='blue', alpha=1)
		plt.gca().add_patch(rect)

	for coord in output:
		rect = patches.Ellipse((coord[0], coord[1]), 10, 10, linewidth=2, facecolor='green', alpha=1)
		plt.gca().add_patch(rect)
	
	plt.savefig("landmark_example.pdf", bbox_inches="tight")


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
	model.load_state_dict(torch.load('../../models/multitask/best_model.pt', map_location=device))

	model.eval()
	model.to(device)
	test_img_idx = 12
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
