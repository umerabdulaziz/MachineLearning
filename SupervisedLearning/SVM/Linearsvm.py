from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(n_samples=100, centers=2, random_state=6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title("Linearly Separable Data (Blobs)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

model = SVC(kernel='linear', C=1) # Hard margin
model.fit(X, y)


w = model.coef_[0]
b = model.intercept_[0]


xline = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # decision boundary line
yline = -(w[0] * xline + b) / w[1]

# Margins: add and subtract 1 from the hyperplane (margin = 1/w)
margin = 1 / np.sqrt(np.sum(w ** 2))
yline_up = yline + margin
yline_down = yline - margin


plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.plot(xline, yline, 'k--')         # decision boundary
plt.plot(xline, yline_up, 'g--')      # positive margin
plt.plot(xline, yline_down, 'g--')    # negative margin
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='black')
plt.title("SVM Decision Boundary with Support Vectors")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

