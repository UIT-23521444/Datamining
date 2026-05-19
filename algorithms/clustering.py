import numpy as np

class KMeansAlgorithm:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def initialize_random_clusters(self, X):
        self.centroids = X[:self.n_clusters].copy().astype(float)
        self.labels = np.zeros(X.shape[0], dtype=int)
        
        return self.labels, self.centroids

    def fit(self, X):
        iteration_result = []
        
        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            
            self.labels = np.argmin(distances, axis=1)
            
            for k in range(self.n_clusters):
                points = X[self.labels == k]
                if len(points) > 0:
                    self.centroids[k] = np.mean(points, axis=0)
            
            self.centroids = np.round(self.centroids, 2)
            
            iteration_result.append({
                "vong_lap": iteration + 1,
                "nhan": self.labels.tolist(),
                "trong_tam": self.centroids.tolist()
            })
            
            if np.all(old_centroids == self.centroids):
                break
                
        return iteration_result, self.centroids, self.labels