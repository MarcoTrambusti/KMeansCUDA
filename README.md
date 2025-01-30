![img.png](img.png)

## Reports
 [Report](https://github.com/MarcoTrambusti/KMeans/blob/main/reports/KMeans_Parallelization_Report.pdf)\
 [Slides](https://github.com/MarcoTrambusti/KMeans/blob/main/reports/PresentationKMeans.pdf)

## K-Means Parallelisation Assignment
### Introduction
This project aims to implement the K-Means algorithm in C++, first in a sequential version and then parallelise it using CUDA to compare and analyse the two versions. The dataset used contains 300147 values relating to the age and total expenditure of customers in a supermarket.

### K-Means algorithm
The K-Means algorithm is a clustering method used to partition a dataset into K distinct clusters. The main steps of the algorithm are:

- **Initialization**: Random selection of K initial centres for the clusters.
- **Assignment of clusters**: Each data point is assigned to the nearest centroid.
- **Update of centroids**: The centroids are recalculated as the average of the points assigned to each cluster.
- **Repetition**: The assignment and update steps are repeated until the centroids change significantly or a maximum number of iterations is reached.

### Implementation
### Representation of Points
Points in the dataset are represented using a Point structure that includes the coordinates, the cluster they belong to and the minimum distance to the cluster.

### Sequential K-Means
The sequential algorithm is divided into two main parts: assignment to clusters and updating of centroids. These steps are repeated for the specified number of iterations.
### Parallelization
Parallelization was implemented using OpenMP. Parallelised steps include:
- Cluster assignment: Calculation of the distance between points and centroids distributed over several threads.
- Updating centroids: Parallel calculation of new centroid coordinates.

### Performance Evaluation
The metrics used to evaluate the performance of the two versions are:

- **Duration**: Average execution time of each configuration
- **Speedup**: Ratio of sequential to parallel execution time
- **Efficiency**: Ratio of Speedup to Number of Threads
