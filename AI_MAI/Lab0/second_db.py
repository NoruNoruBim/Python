import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def plot_first(df):
	cnt = Counter()
	for i in range(1, len(df["SpecialDay"])):
		if df["Revenue"][i] != "FALSE":
			cnt[df["SpecialDay"][i]] += 1
	print(cnt)

	fig, ax = plt.subplots()
	plt.plot(sorted(cnt.keys())[1:], [cnt[key] for key in sorted(cnt.keys())[1:]], 'r')
	ax.set_xlabel("SpecialDay")
	ax.set_ylabel("Number of revenues")
	plt.title("Distribution")
	plt.grid()

	plt.show()

def plot_second(df):
	# There aren't Jan and Apr in the dataset
	d = {"Feb" : Counter(), "Mar" : Counter(), "May" : Counter(), "June" : Counter(), "Jul" : Counter(), "Aug" : Counter(), "Sep" : Counter(), "Oct" : Counter(), "Nov" : Counter(), "Dec" : Counter()}

	for month in d.keys():
		for i in range(1, len(df["VisitorType"])):
			if df["Month"][i] == month:
				d[month][df["VisitorType"][i]] += 1

	months = list(d.keys())

	fig, ax = plt.subplots()
	plt.bar(months, [d[i]["Returning_Visitor"] for i in months], label = "Returning_Visitor")
	plt.bar(months, [d[i]["New_Visitor"] for i in months], label = "New_Visitor")
	plt.grid()
	plt.legend()
	plt.show()
	

tmp = pd.read_csv("C:/hello_world/Python/AI/databases/online_shoppers_intention.csv", header=None)
df = pd.DataFrame({"Administrative" : tmp[0][1:], "Administrative_Duration" : tmp[1][1:], "Informational" : tmp[2][1:],\
					"Informational_Duration" : tmp[3][1:], "ProductRelated" : tmp[4][1:], "ProductRelated_Duration" : tmp[5][1:],\
					"BounceRates" : tmp[6][1:], "ExitRates" : tmp[7][1:], "PageValues" : tmp[8][1:], "SpecialDay" : tmp[9][1:], "Month" : tmp[10][1:],\
					"OperatingSystems" : tmp[11][1:], "Browser" : tmp[12][1:], "Region" : tmp[13][1:], "TrafficType" : tmp[14][1:], "VisitorType" : tmp[15][1:],\
					"Weekend" : tmp[16][1:], "Revenue" : tmp[17][1:]})# now iteration from 1, not 0

plot_first(df)
plot_second(df)
