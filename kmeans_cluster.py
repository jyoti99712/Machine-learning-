from sklearn.cluster import KMeans

# Data: Students marks
X = [[20], [25], [30], [70], [75], [80]]

# Create model (2 groups)
model = KMeans(n_clusters=2, random_state=0)

# Train model
model.fit(X)

# Output groups
print(model.labels_)