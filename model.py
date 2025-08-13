import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
df['Cluster'] = labels

# Train Nearest Neighbor for cluster assignment
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_scaled)

# Save everything
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('dbscan.pkl', 'wb') as f:
    pickle.dump(dbscan, f)

with open('nn.pkl', 'wb') as f:
    pickle.dump(nn, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)