import matplotlib.pyplot as plt
import pandas as pd
import os


def read_data(file):
	atts = []
	scores = []
	lines = []
	with open(file, "r") as f:
		for line in f.readlines()[2:-5]:
			line = line.strip().split(" ")
			line = [x for x in line if x]
			lines.append(line)
			atts.append(line[0].replace("_", " "))
			scores.append(float(line[-1]))
	return atts, scores, lines


def latex_table():
	_, _, scores_single = read_data(os.path.join("../../models", "attributesSGD", "results.txt"))
	_, _, scores_multi = read_data(os.path.join("../../models", "multitask", "results_attr.txt"))
	sep = " & "
	end = " \\\\"

	for j in range(len(scores_single)):
		str_ = r"\multicolumn{1}{|l|}{" + scores_single[j][0].replace("_", " ") + "}" + sep
		for i in range(1, 5):
			str_ += scores_single[j][i] + sep + scores_multi[j][i] + (sep if i < 4 else "")
		print(str_ + end)


def plot():
	atts, scores_single, _ = read_data(os.path.join("../../models", "attributesSGD", "results.txt"))
	_, scores_multi, _ = read_data(os.path.join("../../models", "multitask", "results_attr.txt"))

	plt.rcParams["figure.figsize"] = (15, 5)
	colors = ["lightskyblue", "royalblue"]
	df = pd.DataFrame({"Single Task": scores_single, "Multi Task": scores_multi}, columns=['Single Task', 'Multi Task'],
	                  index=atts)
	df.plot(kind='bar', width=0.7, color=colors, zorder=3)
	plt.grid(color='#95a5a6', linestyle='-', linewidth=0.8, alpha=0.2, zorder=-1)
	plt.xlabel("Attribute")
	plt.ylabel("F1-score")

	bal_idx = [3, 19, 20, 21, 22, 32, 37]

	for x in bal_idx:
		plt.gca().get_xticklabels()[x - 1].set_bbox(dict(facecolor="lightgray"))

	plt.xticks(
		rotation=50,
		horizontalalignment='right',
		fontweight='light',
		fontsize='medium',
	)

	plt.title("F1-score comparison on Test set")
	plt.savefig("bars.pdf", bbox_inches="tight")
	plt.show()


plot()
latex_table()
