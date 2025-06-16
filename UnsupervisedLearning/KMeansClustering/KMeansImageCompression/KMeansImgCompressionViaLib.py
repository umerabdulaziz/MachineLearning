from sklearn.cluster import KMeans
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt

original_img = imread('bird_small.png')
img_shape = original_img.shape

X = original_img.reshape(-1, img_shape[2])

kmeans = KMeans(n_clusters=16, max_iter=10, n_init=1)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
idx = kmeans.predict(X)
compressed_img = centroids[idx].reshape(img_shape)

plt.imshow(compressed_img)
plt.axis('off')
plt.show()
