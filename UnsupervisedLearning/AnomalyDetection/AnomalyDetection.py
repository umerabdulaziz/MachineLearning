import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
import os

os.makedirs("data", exist_ok=True)

# Low-dimensional (2D) dataset
np.random.seed(0)
X_train = np.random.multivariate_normal([14, 15], [[1.8, 0.5], [0.5, 1.7]], 307)
X_val_normal = np.random.multivariate_normal([14, 15], [[1.8, 0.5], [0.5, 1.7]], 295)
X_val_anomalies = np.random.multivariate_normal([30, 30], [[1.0, 0], [0, 1.0]], 12)
X_val = np.vstack([X_val_normal, X_val_anomalies])
y_val = np.hstack([np.zeros(295), np.ones(12)])

# High-dimensional dataset (11 features)
X_train_high = np.random.normal(loc=0, scale=1, size=(1000, 11))
X_val_high = np.random.normal(loc=0, scale=1, size=(90, 11))
anomalies_high = np.random.uniform(low=-10, high=10, size=(10, 11))
X_val_high = np.vstack([X_val_high, anomalies_high])
y_val_high = np.hstack([np.zeros(90), np.ones(10)])

np.save("data/X_train.npy", X_train)
np.save("data/X_val.npy", X_val)
np.save("data/y_val.npy", y_val)
np.save("data/X_train_high.npy", X_train_high)
np.save("data/X_val_high.npy", X_val_high)
np.save("data/y_val_high.npy", y_val_high)


def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var

def multivariate_gaussian(X, mu, var):
    cov = np.diag(var)
    dist = multivariate_normal(mean=mu, cov=cov)
    return dist.pdf(X)

def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_f1 = 0
    for epsilon in np.linspace(min(p_val), max(p_val), 1000):
        preds = p_val < epsilon
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1

X_train = np.load("data/X_train.npy")
X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")

mu, var = estimate_gaussian(X_train)
p_train = multivariate_gaussian(X_train, mu, var)
p_val = multivariate_gaussian(X_val, mu, var)

epsilon, f1 = select_threshold(y_val, p_val)
print("[2D] Best epsilon:", epsilon)
print("[2D] Best F1 score:", f1)

anomalies = X_val[p_val < epsilon]
# Create meshgrid for contours
x = np.linspace(min(X_train[:, 0])-5, max(X_train[:, 0])+5, 100)
y = np.linspace(min(X_train[:, 1])-5, max(X_train[:, 1])+5, 100)
X_mesh, Y_mesh = np.meshgrid(x, y)
Z = multivariate_normal(mean=mu, cov=np.diag(var)).pdf(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))
Z = Z.reshape(X_mesh.shape)

# Contour plot of the Gaussian distribution
plt.contour(X_mesh, Y_mesh, Z, levels=10, cmap="Blues")
plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0], X_train[:,1], label="Training Data")
plt.scatter(anomalies[:,0], anomalies[:,1], c='r', marker='x', label="Anomalies")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("2D Gaussian Anomaly Detection")
plt.legend()
plt.tight_layout()
plt.savefig("2d_anomaly_plot.png")
plt.show()

X_train_high = np.load("data/X_train_high.npy")
X_val_high = np.load("data/X_val_high.npy")
y_val_high = np.load("data/y_val_high.npy")

def high_dim_anomaly_detection(X_train, X_val, y_val):
    mu = np.mean(X_train, axis=0)
    var = np.var(X_train, axis=0)
    cov = np.diag(var)
    dist = multivariate_normal(mean=mu, cov=cov)
    p_val = dist.pdf(X_val)
    epsilon, f1 = select_threshold(y_val, p_val)
    print("[High-Dim] Best epsilon:", epsilon)
    print("[High-Dim] Best F1 score:", f1)
    return epsilon, f1

epsilon_hd, f1_hd = high_dim_anomaly_detection(X_train_high, X_val_high, y_val_high)

print("\n✅ All anomaly detection projects complete.")