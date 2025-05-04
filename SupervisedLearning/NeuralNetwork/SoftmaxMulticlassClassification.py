import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
# Split for demonstration purposes
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)


np.random.seed(42)
n_classes = 3
n_features = X.shape[1]
weights = np.random.randn(n_features, n_classes)  # Shape: (2, 3)
biases = np.random.randn(1, n_classes)            # Shape: (1, 3)


def softmax(logits):
    """Applies softmax activation to raw scores (logits)."""
    exp_values = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stability trick
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

logits = np.dot(X_train, weights) + biases  # Shape: (210, 3)

probabilities = softmax(logits)

np.set_printoptions(precision=3, suppress=True)
print("Example probabilities after softmax (first 5 samples):\n", probabilities[:5])

predictions = np.argmax(probabilities, axis=1)

accuracy = np.mean(predictions == y_train)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=predictions, cmap='viridis', edgecolor='k', alpha=0.7)
plt.title("Softmax Multi-class Classification (Train Data)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Predicted Class')
plt.grid(True)
plt.tight_layout()
plt.show()