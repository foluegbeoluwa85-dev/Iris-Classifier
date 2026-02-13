import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
iris = load_iris()
X = iris.data        # shape (150, 4)
y = iris.target      # shape (150,)

print(iris.feature_names, iris.target_names)

# -------------------------------------------------------
# Train/Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------------
# Decision Tree Classifier
# -------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Predictions:", y_pred_dt[:5])
print("True labels:", y_test[:5])

dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", dt_accuracy)

# -------------------------------------------------------
# k-NN Classifier
# -------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("k-NN Accuracy:", knn_accuracy)

# -------------------------------------------------------
# Confusion Matrix (Decision Tree)
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_dt)
target_names = iris.target_names

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Iris Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
joblib.dump(dt_model, "outputs/model.joblib")
print("Saved: outputs/model.joblib")
