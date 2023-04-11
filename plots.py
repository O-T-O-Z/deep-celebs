import matplotlib.pyplot as plt
import json
import numpy as np
import os


def load_data():
	models = ["attributes", "landmarks", "multitask"]
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
		ax[i].plot(np.log(model["train_loss"]), label="Training", color=colors[0])
		ax[i].plot(np.log(model["eval_loss"]), label="Validation", color=colors[1])
		ax[i].set_ylabel("log " + losses[i] + " Loss")
		ax[i].set_xlabel("Epoch")
		ax[i].set_title(titles[i] + " Task", fontsize=11)
		ax[i].grid(color='#95a5a6', linestyle='-', linewidth=0.8, alpha=0.2)

	handles, labels = ax[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='center right')

	plt.suptitle("Learning Curves", fontsize=15)
	plt.savefig("loss.pdf", bbox_inches="tight")

	plt.show()


plot_data()
