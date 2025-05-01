import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, flip_y=0, random_state=42)
y = y.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, weights, lambda_reg):
    m = len(y)
    predictions = sigmoid(X.dot(weights))
    cost = -1/m * np.sum(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))
    reg_term = lambda_reg / (2 * m) * np.sum(weights[1:] ** 2)
    return cost + reg_term

def compute_gradient(X, y, weights, lambda_reg):
    m = len(y)
    predictions = sigmoid(X.dot(weights))
    error = predictions - y
    gradient = (1 / m) * X.T.dot(error)
    gradient[1:] += (lambda_reg / m) * weights[1:]  # L2 regularization
    return gradient

def gradient_descent(X, y, learning_rate=0.1, epochs=1000, lambda_reg=0.1):
    m, n = X.shape
    X_b = np.hstack((np.ones((m, 1)), X))  # Add bias term
    weights = np.zeros((n + 1, 1))

    for i in range(epochs):
        gradient = compute_gradient(X_b, y, weights, lambda_reg)
        weights -= learning_rate * gradient

        if i % 100 == 0:
            loss = compute_loss(X_b, y, weights, lambda_reg)
            print(f"Epoch {i} - Loss: {loss:.4f}")
    return weights

def predict(X, weights):
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    return (sigmoid(X_b.dot(weights)) >= 0.5).astype(int)

def plot_decision_boundary(X, y, weights):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(weights[0] + weights[1]*x1) / weights[2]

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y.ravel() == 0][:, 0], X[y.ravel() == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y.ravel() == 1][:, 0], X[y.ravel() == 1][:, 1], color='blue', label='Class 1')
    plt.plot(x1, x2, color='green', label='Decision Boundary')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.show()

weights = gradient_descent(X_train, y_train, learning_rate=0.1, epochs=1000, lambda_reg=0.5)
y_pred = predict(X_test, weights)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
plot_decision_boundary(X_test, y_test, weights)
