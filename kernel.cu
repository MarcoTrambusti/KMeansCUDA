#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <filesystem>
#include <omp.h>
#include <cuda_runtime.h>
#include <random>
#include "common/common.h"

struct Points {
    std::vector<float> flat_coord;
    std::vector<int> clusters;

    Points(const int numPoints, const std::vector<Point>& points) : flat_coord(numPoints*2), clusters(numPoints,-1) {
        for (int i=0; i<points.size(); i++) {
            flat_coord[i * 2] = points[i].x;
            flat_coord[i * 2 + 1] = points[i].y;
        }
    }
};

__global__ void kmeans_kernel(const float* d_datapoints, int* d_clust_assn, const float* d_centroids, const int numPoints, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        float minDist = FLT_MAX;
        int bestCluster = -1;
        float px = d_datapoints[2 * idx];
        float py = d_datapoints[2 * idx + 1];

        for (int c = 0; c < k; ++c) {
            float cx = d_centroids[2 * c];
            float cy = d_centroids[2 * c + 1];
            float dist = (px - cx) * (px - cx) + (py - cy) * (py - cy);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }
        d_clust_assn[idx] = bestCluster;
    }
}

__global__ void update_centroids(const float* d_datapoints, const int* d_clust_assn, float* d_centroids, int* d_clust_sizes, int numPoints, int k) {
    extern __shared__ float shared_mem[];
    float* shared_centroids = shared_mem;
    auto shared_sizes = reinterpret_cast<int *>(&shared_centroids[2 * k]);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (tid < 2 * k) {
        shared_centroids[tid] = 0.0f;
    }
    if (tid < k) {
        shared_sizes[tid] = 0;
    }
    __syncthreads();

    if (idx < numPoints) {
        int cluster = d_clust_assn[idx];
        atomicAdd(&shared_centroids[2 * cluster], d_datapoints[2 * idx]);
        atomicAdd(&shared_centroids[2 * cluster + 1], d_datapoints[2 * idx + 1]);
        atomicAdd(&shared_sizes[cluster], 1);
    }
    __syncthreads();

    if (tid < 2 * k) {
        atomicAdd(&d_centroids[tid], shared_centroids[tid]);
    }
    if (tid < k) {
        atomicAdd(&d_clust_sizes[tid], shared_sizes[tid]);
    }
}

void kmeans_cuda(std::vector<Point>& points, const std::vector<Point>& centroids, int k, int epochs, int numThreads) {
    int numPoints = points.size();
    int* clust_assn = new int[numPoints]; // Cluster assignments

    float* d_datapoints;
    int* d_clust_assn;
    float* d_centroids;
    int* d_clust_sizes;

    // Allocate memory on GPU
    cudaMalloc(&d_datapoints, numPoints * 2 * sizeof(float));
    cudaMalloc(&d_clust_assn, numPoints * sizeof(int));
    cudaMalloc(&d_centroids, k * 2 * sizeof(float));
    cudaMalloc(&d_clust_sizes, k * sizeof(int));

    // Copy points to d_datapoints
    auto pointsSoA = Points(numPoints, points);
    cudaMemcpy(d_datapoints, pointsSoA.flat_coord.data(), numPoints * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Copy centroids to d_centroids
    auto centroidsSoA = Points(k, centroids);

    cudaMemcpy(d_centroids, centroidsSoA.flat_coord.data(), k * 2 * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (numPoints + numThreads - 1) / numThreads;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Assign points to clusters
        kmeans_kernel<<<numBlocks, numThreads>>>(d_datapoints, d_clust_assn, d_centroids, numPoints, k);
        cudaDeviceSynchronize();

        // Reset centroids and sizes
        cudaMemset(d_centroids, 0, k * 2 * sizeof(float));
        cudaMemset(d_clust_sizes, 0, k * sizeof(int));

        // Update centroids
        update_centroids<<<numBlocks, numThreads, k * 2 * sizeof(float) + k * sizeof(int)>>>(d_datapoints, d_clust_assn, d_centroids, d_clust_sizes, numPoints, k);
        cudaDeviceSynchronize();

        // Normalize centroids
        std::vector<float> h_centroids(k * 2);
        std::vector<int> h_clust_sizes(k);
        cudaMemcpy(h_centroids.data(), d_centroids, k * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_clust_sizes.data(), d_clust_sizes, k * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < k; ++i) {
            if (h_clust_sizes[i] > 0) {
                h_centroids[2 * i] /= h_clust_sizes[i];
                h_centroids[2 * i + 1] /= h_clust_sizes[i];
            }
        }
        cudaMemcpy(d_centroids, h_centroids.data(), k * 2 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Copy cluster assignments back to host
    cudaMemcpy(pointsSoA.clusters.data(), d_clust_assn, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Update points with new cluster assignments
    for (int i = 0; i < numPoints; ++i) {
        points[i].cluster = pointsSoA.clusters[i];
    }

    // Free memory
    delete[] clust_assn;
    cudaFree(d_datapoints);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids);
    cudaFree(d_clust_sizes);
}

int main() {
    std::vector<int> numThreadsList = {1,64, 128, 256, 512}; // Include 128, 256, 512, 1024 threads per block
    std::vector<int> numPointsList = {500, 7000, 20000, 50000, 126000}; // List of number of points
    std::vector<std::tuple<int, int, double, double, double>> results;
    std::unordered_map<int, int> pointsToClusters = { {500, 5}, {7000, 10}, {20000, 12}, {50000, 15}, {126000, 18}, {301487, 20} };
    std::unordered_map<int, int> pointsToEpochs = { {500, 20}, {7000, 50}, {20000, 70}, {50000, 100}, {126000, 120}, {301487, 200} };

    float minX, minY, maxX, maxY;
    std::vector<Point> points = KMeansHelper::readAndNormalizeCSV("resources/a.csv", minX, minY, maxX, maxY);
    numPointsList.push_back(points.size());

    for (int numPoints : numPointsList) {
        int numClusters = pointsToClusters[numPoints];
        std::cout << "Number of points: " << numPoints << " -> Number of clusters: " << numClusters << std::endl;
        double avgDurationSequential = 0.0;
        std::vector<Point> centroids;
        KMeansHelper::init_centroids(numClusters, points, centroids);

        for (int numThreads : numThreadsList) {
            cudaDeviceReset();
            int numMeasurements = 5;
            double totalDuration = 0.0;

            for (int measurement = 0; measurement < numMeasurements; ++measurement) {
                std::vector<Point> pointsCopy(points.begin(), points.begin() + numPoints);
                if (measurement == 0) {
                    KMeansHelper::draw_chart_gnu(pointsCopy, "initial_points_number_" + std::to_string(numPoints) + ".png", minX, minY, maxX, maxY, numClusters);
                }
                double startParallel = omp_get_wtime();
                if (numThreads == 1) {
                    KMeansHelper::kmeans_sequential(pointsCopy, centroids, numClusters, pointsToEpochs[numPoints]);
                } else {
                    kmeans_cuda(pointsCopy, centroids, numClusters, pointsToEpochs[numPoints], numThreads);
                }
                double endParallel = omp_get_wtime();
                totalDuration += (endParallel - startParallel);

                if (measurement == 0) {
                    KMeansHelper::draw_chart_gnu(pointsCopy, "kmeans_numThreads_" + std::to_string(numThreads) + "_numPoints_" + std::to_string(numPoints) + "_measurement_n_" + std::to_string(measurement) + ".png", minX, minY, maxX, maxY, numClusters);
                }
            }

            if (numThreads == 1) {
                avgDurationSequential = totalDuration / numMeasurements;
            }

            double avgDurationParallel = totalDuration / numMeasurements;
            double speedup = avgDurationSequential / avgDurationParallel;
            double efficiency = speedup / numThreads;

            std::cout << "Points: " << numPoints << ", Threads per block: " << numThreads
                      << ", Avg Duration (Parallel): " << avgDurationParallel << "s, Speedup: " << speedup
                      << ", Efficiency: " << efficiency << std::endl;

            results.emplace_back(numPoints, numThreads, avgDurationParallel, speedup, efficiency);
        }
    }

    KMeansHelper::logExecutionDetails("performance_log.csv", results);

    return 0;
}