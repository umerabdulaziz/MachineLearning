import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


digits = load_digits()
X, y = digits.data, digits.target

mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]
y = y.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def my_dense_v(A_in, W, b, g):
    z = np.matmul(A_in, W) + b
    return g(z)

# Define network layers: (Input 64 -> 25 -> 15 -> 1)
np.random.seed(42)
W1 = np.random.randn(64, 25)
b1 = np.zeros((1, 25))

W2 = np.random.randn(25, 15)
b2 = np.zeros((1, 15))

W3 = np.random.randn(15, 1)
b3 = np.zeros((1, 1))

def model_forward(X):
    a1 = my_dense_v(X, W1, b1, sigmoid)
    a2 = my_dense_v(a1, W2, b2, sigmoid)
    a3 = my_dense_v(a2, W3, b3, sigmoid)
    return a3

def predict(X):
    return (model_forward(X) >= 0.5).astype(int)

yhat_train = predict(X_train)
yhat_test = predict(X_test)

train_acc = np.mean(yhat_train == y_train)
test_acc = np.mean(yhat_test == y_test)

print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy:  {test_acc:.2f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

yhat_all = predict(X)

plt.figure(figsize=(8,6))
for label in [0, 1]:
    plt.scatter(
        X_pca[(y.flatten() == label) & (yhat_all.flatten() == label), 0],
        X_pca[(y.flatten() == label) & (yhat_all.flatten() == label), 1],
        label=f"Correct {label}",
        alpha=0.6,
    )
    plt.scatter(
        X_pca[(y.flatten() == label) & (yhat_all.flatten() != label), 0],
        X_pca[(y.flatten() == label) & (yhat_all.flatten() != label), 1],
        label=f"Misclassified {label}",
        marker='x',
        color='red'
    )

plt.title("Manual Neural Network Classification (0 vs 1)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
