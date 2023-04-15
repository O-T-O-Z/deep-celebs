import matplotlib.pyplot as plt
import json
import numpy as np
import os


def load_data():
	models = ["attributesSGD", "landmarksSGD", "multitask_with_loss"]
	model_data = []

	for m in models:
		with open(os.path.join("models", m, "metrics.json"), "r") as f:
			model_data.append(json.load(f))

	return model_data


def plot_data():
	model_data = load_data()

	fig, ax = plt.subplots(1, 3, figsize=(15, 5))

	losses = ["BCE", "MSE", "(BCE + MSE)"]
	titles = ["Attribute", "Landmark", "Attribute + Landmark"]
	colors = ["orangered", "limegreen"]

	for i, model in enumerate(model_data):
		y_1 = model["train_loss"][1:]
		y_2 = model["eval_loss"][1:]
		ax[i].plot(range(2, len(y_1) + 2), y_1, label="Training", color=colors[0])
		ax[i].plot(range(2, len(y_2) + 2), y_2, label="Validation", color=colors[1])
		ax[i].set_ylabel(losses[i] + " Loss")
		ax[i].set_xlabel("Epoch")
		ax[i].set_title(titles[i] + " Task", fontsize=11)
		ax[i].grid(color='#95a5a6', linestyle='-', linewidth=0.8, alpha=0.2)
		if i == 0 or i == 2:
			ax[i].set_ylim(0, 0.7)
		else:
			ax[i].set_ylim(0, 11)
			ax[i].set_xticks(np.arange(2, 52, 8.0))

	handles, labels = ax[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncols=2,bbox_to_anchor=(0.5, -0.08))

	plt.suptitle("Learning Curves", fontsize=15)
	plt.savefig("loss.pdf", bbox_inches="tight")

	plt.show()


plot_data()
