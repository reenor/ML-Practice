# https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

## Data Example
iris = load_iris()
X = iris['data']
y = iris['target']

n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

## To get a feeling for how features (independent variables) are related, let us visualize them via histograms and scatter plots.
fig, ax = plt.subplots(nrows=n_features, ncols=n_features, figsize= (8, 8))
fig.tight_layout()

names = iris.feature_names

for i, j in zip(*np.triu_indices_from(ax, k=1)):
    ax[j, i].scatter(X[:, j], X[:, i], c = y)
    ax[j, i].set_xlabel(names[j])
    ax[j, i].set_ylabel(names[i])
    ax[i, j].set_axis_off()

for i in range(n_features):
    ax[i, i].hist(X[:, i], color = 'lightblue')
    ax[i, i].set_ylabel('Count')
    ax[i, i].set_xlabel(names[i])

plt.show()

## PCA with the covariance method
# Step 1: Standardize the data
# We can standardize features by removing the mean and scaling to unit variance.
def mean(X): # np.mean(X, axis = 0)  
    return sum(X)/len(X)  

def sd(X): # np.std(X, axis = 0)
    return (sum((X - mean(X))**2)/len(X))**0.5

def standardize_data(X):
    return (X - mean(X))/sd(X)

print('Mean of X:', mean(X));
print('SD of X:', sd(X));
X_std = standardize_data(X)

# print("X ", X)
# print("X_std ", X_std)

# Step 2: Find the covariance matrix
# The covariance matrix of standardized data can be calculated as follows.
def covariance(X): 
    return (X.T @ X)/(X.shape[0]-1)

cov_mat = covariance(X_std) # np.cov(X_std.T)
print('Cov matrix ', cov_mat)

# Step 3: Find the eigenvectors and eigenvalues of the covariance matrix
from numpy.linalg import eig

# Eigen decomposition of covariance matrix
eig_vals, eig_vecs = eig(cov_mat)

# Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
eig_vecs = eig_vecs*signs[np.newaxis,:]
eig_vecs = eig_vecs.T

print('Eigenvalues \n', eig_vals)
print('Eigenvectors \n', eig_vecs)

# Step 4: Rearrange the eigenvectors and eigenvalues
# Here, we sort eigenvalues in descending order.

# We first make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i,:]) for i in range(len(eig_vals))]

# Then, we sort the tuples from the highest to the lowest based on eigenvalues magnitude
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# For further usage
eig_vals_sorted = np.array([x[0] for x in eig_pairs])
eig_vecs_sorted = np.array([x[1] for x in eig_pairs])

print(eig_pairs)

# Step 5: Choose principal components
# Now, we choose the first k eigenvectors where k is the number of dimensions of the new feature subspace (k ≤ nfeatures).
k = 2
W = eig_vecs_sorted[:k, :] # Projection matrix

#print(W.shape)

# Note that, the value of k can be set in a wise way through explained variance.
# The explained variance tells us how much information (variance) can be attributed to each of the principal components.
eig_vals_total = sum(eig_vals)
explained_variance = [(i / eig_vals_total)*100 for i in eig_vals_sorted]
explained_variance = np.round(explained_variance, 2)
cum_explained_variance = np.cumsum(explained_variance)

print('Explained variance: {}'.format(explained_variance))
print('Cumulative explained variance: {}'.format(cum_explained_variance))

# plt.plot(np.arange(1,n_features+1), cum_explained_variance, '-o')
# plt.xticks(np.arange(1,n_features+1))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance');
# plt.show()

# Step 6: Project the data
# Finally, we can transform the data X via the projection matrix W to obtain a k-dimensional feature subspace.
X_proj = X_std.dot(W.T)

print(X_proj.shape)

# Here, we visualize the transformed data in PCA space of the first two PCs: PC1 and PC2.
# plt.scatter(X_proj[:, 0], X_proj[:, 1], c = y)
# plt.xlabel('PC1'); plt.xticks([])
# plt.ylabel('PC2'); plt.yticks([])
# plt.title('2 components, captures {} of total variation'.format(cum_explained_variance[1]))
# plt.show()