from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)

print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)