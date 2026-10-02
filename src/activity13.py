from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "pruning_team4_biomedical.csv"
data = pd.read_csv(data_path)
print("Data loaded successfully.")

X = data.drop(columns="cardiac_crisis_alert")
y = data["cardiac_crisis_alert"]

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.25,
	random_state=42,
	stratify=y,
)

tree_candidates = [
	("α_1 (Unconstrained)", 0.001),
	("α_2 (Optimal Pruned)", 0.015),
	("α_3 (Over-Pruned)", 0.080),
]
model_results = []

for candidate_name, ccp_alpha in tree_candidates:
	clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
	clf.fit(X_train, y_train)

	training_error = 1.0 - clf.score(X_train, y_train)
	test_error = 1.0 - clf.score(X_test, y_test)
	leaf_count = clf.get_n_leaves()
	max_depth = clf.get_depth()
	total_cost = test_error + ccp_alpha * leaf_count
	model_results.append(
		{
			"Candidate α": candidate_name,
			"ccp_alpha": ccp_alpha,
			"Value": total_cost,
			"Leaf Count(|T|)": leaf_count,
			"Max Depth(D_max)": max_depth,
			"Train Error(R_train)": training_error,
			"Test Error(R_test)": test_error,
		}
	)
	print(
		f"{candidate_name} (ccp_alpha={ccp_alpha:.3f})\n"
		f"  Leaves: {leaf_count}\n"
		f"  Depth: {max_depth}\n"
		f"  Training error: {training_error:.4f}\n"
		f"  Test error: {test_error:.4f}\n"
		f"  Total cost: {total_cost:.4f}\n"
	)

minimum_cost_model = min(model_results, key=lambda result: result["Value"])
lowest_test_error_model = min(model_results, key=lambda result: result["Test Error(R_test)"])
print(
	f"Minimum total cost: ccp_alpha={minimum_cost_model['ccp_alpha']:.3f}, "
	f"R_alpha(T)={minimum_cost_model['Value']:.4f}"
)
print(
	f"Lowest test error: ccp_alpha={lowest_test_error_model['ccp_alpha']:.3f}, "
	f"R_test(T)={lowest_test_error_model['Test Error(R_test)']:.4f}"
)

summary_table = pd.DataFrame(model_results)
print("\nModel Summary Table (Value = R_test + ccp_alpha × |T|):")
print(
	summary_table.to_string(
		index=False,
		formatters={
			"ccp_alpha": "{:.3f}".format,
			"Value": "{:.4f}".format,
			"Leaf Count(|T|)": "{:d}".format,
			"Max Depth(D_max)": "{:d}".format,
			"Train Error(R_train)": lambda error: f"{error * 100:.2f}%",
			"Test Error(R_test)": lambda error: f"{error * 100:.2f}%",
		},
	)
)
