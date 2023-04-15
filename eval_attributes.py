import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from tqdm import tqdm
from tabulate import tabulate
from multi_task import MultiTask
from attributes import AttributeClassifier
from utils import calculate_f1, calculate_cm


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	test_data = CelebA(root='data', split="test", download=False, transform=transform, target_type=["attr"])
	test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8)
	# model = AttributeClassifier(n_classes=40)
	model = MultiTask()

	model.load_state_dict(torch.load('models/multitaskSGD/best_model.pt', map_location=device))

	model.eval()
	model.to(device)
	with torch.no_grad():
		n_class_correct = {arg: 0 for arg in test_data.attr_names if arg}
		attribute_metrics = {attr: {"TP": 0, "FP": 0, "FN": 0} for attr in test_data.attr_names}
		n_samples = 0
		all_eval_preds = []
		all_eval_labels = []

		for inputs, labels in tqdm(test_data_loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = torch.round(model(inputs)[0])
			all_eval_preds.append(torch.round(outputs))
			all_eval_labels.append(labels)
			for i, out in enumerate(outputs):
				for j, label in enumerate(labels[i]):
					if out[j] == label:
						n_class_correct[test_data.attr_names[j]] += 1
				n_samples += 1
			attribute_metrics = calculate_cm(torch.round(outputs), labels, attribute_metrics)
	metrics = calculate_f1(attribute_metrics)
	n_class_portion = {
		arg: [n_class_correct[arg] / n_samples, metrics[arg]["precision"], metrics[arg]["recall"], metrics[arg]["f1"]]
		for arg in test_data.attr_names if arg}
	headers = ["Attribute", "Accuracy", "Precision", "Recall", "F1"]
	print(tabulate([[k, str(round(v[0], 2)), str(round(v[1], 2)), str(round(v[2], 2)), str(round(v[3], 2))] for k, v in
	                n_class_portion.items()], headers=headers))
	print("Overall Accuracy: ", sum(n_class_correct.values()) / (n_samples * 40))
	print("Overall Precision: ", sum([metrics[arg]["precision"] for arg in metrics.keys()]) / 40)
	print("Overall Recall: ", sum([metrics[arg]["recall"] for arg in metrics.keys()]) / 40)
	print("Overall F1: ", sum([metrics[arg]["f1"] for arg in metrics.keys()]) / 40)
	print("Number of samples: ", n_samples)


if __name__ == "__main__":
	main()
