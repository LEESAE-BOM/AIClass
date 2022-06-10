import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0) #4 개의 클러스터 생성
plt.scatter(X[:, 0], X[:, 1], s=50);
plt.show()
plt.cla()
n_cluster = range(1,10)

kmeans_test = [KMeans(n_clusters=i) for i in n_cluster]

score = [kmeans_test[i].fit(X).inertia_ for i in range(len(kmeans_test))]

plt.plot(n_cluster,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
plt.cla()
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
# c: 마커의 색상, s: 마커의 크기, alpha: 색상의 투명도 (0: 완전투명, 1:완전불투명)
plt.show()