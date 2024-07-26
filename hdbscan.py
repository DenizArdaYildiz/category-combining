import json
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the JSON file
with open('embeddings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract category names and vectors
category_names = list(data.keys())
vectors = list(data.values())

# Sanitize category names to keep Turkish characters
def sanitize_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

category_names = [sanitize_text(name) for name in category_names]

# Convert the list of vectors to a numpy array
vectors = np.array(vectors)

# Create a UMAP instance and fit-transform the data
umap_reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', min_dist=0.1)
umap_embedding = umap_reducer.fit_transform(vectors)

# Scale the UMAP embedding to increase the distance between points
scaling_factor = 10  # Adjust this factor to increase/decrease the spread
umap_embedding *= scaling_factor

# Create an HDBSCAN instance with initial parameters from scikit-learn
hdbscan_clusterer = HDBSCAN(min_cluster_size=5, metric='euclidean')
cluster_labels = hdbscan_clusterer.fit_predict(umap_embedding)

# Create a dictionary to store elements of each cluster
clusters = defaultdict(list)

# Iterate through labels and store elements
for idx, label in enumerate(cluster_labels):
    clusters[label].append(category_names[idx])

# Post-processing to ensure each cluster contains around 5 elements
final_clusters = defaultdict(list)
remaining_elements = []

for cluster_id, elements in clusters.items():
    while len(elements) > 5:
        final_clusters[cluster_id].extend(elements[:5])
        remaining_elements.extend(elements[5:])
        elements = elements[5:]
    if elements:
        final_clusters[cluster_id].extend(elements)

# Attempt to distribute remaining elements across clusters
remaining_clusters = list(final_clusters.keys())
for elem in remaining_elements:
    if remaining_clusters:
        cluster_id = remaining_clusters.pop(0)
        if len(final_clusters[cluster_id]) < 5:
            final_clusters[cluster_id].append(elem)
        else:
            remaining_clusters.append(cluster_id)
    else:
        # If no clusters are available, create a new cluster
        cluster_id = max(final_clusters.keys(), default=0) + 1
        final_clusters[cluster_id] = [elem]

# Convert final clusters dictionary to a JSON serializable format
clusters_dict = {str(cluster_id): elements for cluster_id, elements in final_clusters.items()}

# Save clusters to a JSON file
with open('clusters.json', 'w', encoding='utf-8') as f:
    json.dump(clusters_dict, f, indent=4, ensure_ascii=False)

# Print the clusters and their elements (optional)
for cluster_id, elements in clusters_dict.items():
    print(f"Cluster {cluster_id}:")
    for element in elements:
        print(f" - {element}")
    print()  # New line for better readability

# Plot without noise and text
plt.figure(figsize=(12, 8))
unique_labels = set(cluster_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Skip noise points
        continue

    class_member_mask = (cluster_labels == k)

    # Plot core samples
    xy = umap_embedding[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

# Create a scatter plot for adding the color bar with smaller marker size
scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=cluster_labels, cmap='Spectral', s=10)

# Add a color bar
plt.colorbar(scatter)

# Display the plot
plt.title("UMAP + HDBSCAN Clustering of Category Embeddings (No Noise, No Labels)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.show()
