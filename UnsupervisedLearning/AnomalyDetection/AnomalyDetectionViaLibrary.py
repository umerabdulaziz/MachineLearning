
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score

X_train = np.load("data/X_train.npy")
X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")


def lib_based_anomaly_detection(X_train, X_val, y_val):
    model = EllipticEnvelope(contamination=0.03)
    model.fit(X_train)
    preds = model.predict(X_val)
    preds = (preds == -1).astype(int)  # convert -1 to 1 (anomaly), 1 to 0 (normal)
    f1 = f1_score(y_val, preds)
    print("[Lib-Based] F1 Score:", f1)
    return preds

preds = lib_based_anomaly_detection(X_train, X_val, y_val)

plt.figure(figsize=(8, 6))
plt.scatter(X_val[:, 0], X_val[:, 1], c=preds, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Anomaly Detection (Library-based)")
plt.tight_layout()
plt.savefig("lib_anomaly_plot.png")
plt.show()
