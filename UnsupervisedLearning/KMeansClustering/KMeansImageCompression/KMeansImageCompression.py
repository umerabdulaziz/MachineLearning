import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def find_closest_centroids(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]
        if len(points) == 0:
            centroids[k] = X[np.random.randint(0, m)]
        else:
            centroids[k] = np.mean(points, axis=0)
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10):
    centroids = initial_centroids
    for i in range(max_iters):
        print(f"K-Means iteration {i+1}/{max_iters}")
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, centroids.shape[0])
    return centroids, idx

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    return X[randidx[:K]]

original_img = imread('bird_small.png')
print("Image shape:", original_img.shape)
if original_img.shape[-1] == 4:
    original_img = original_img[..., :3]
if original_img.dtype == np.uint8:
    original_img = original_img.astype(float) / 255
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

K = 16
max_iters = 10
initial_centroids = kMeans_init_centroids(X_img, K)
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

X_recovered = centroids[idx]
X_recovered = np.reshape(X_recovered, original_img.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(original_img)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(X_recovered)
ax[1].set_title(f"Compressed Image with {K} colors")
ax[1].axis('off')

plt.show()
