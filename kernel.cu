#include <iostream>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <omp.h>
#include <cuda_runtime.h>
#include <random>
#include "common/common.h"
#include <vector>
#include <thrust/device_vector.h>

struct Points {
    std::vector<float> flat_coord;
    std::vector<int> clusters;

    Points(const int numPoints, const std::vector<Point> &points) : flat_coord(numPoints * 2), clusters(numPoints, -1) {
        for (int i = 0; i < points.size(); i++) {
            flat_coord[i * 2] = points[i].x;
            flat_coord[i * 2 + 1] = points[i].y;
        }
    }
};

constexpr int POINTS_PER_THREAD   = 10;
constexpr int CLUSTERS_PER_THREAD = 5;

__global__ void assignClusters(const float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_nPoints, float *d_sumX, float *d_sumY, int numPoints, const int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_centroids[];
    int *s_nPoints = reinterpret_cast<int *>(&s_centroids[2 * k]);
    float *s_sumX = reinterpret_cast<float *>(s_nPoints + k);
    float *s_sumY = s_sumX + k;

    if (threadIdx.x < k) {
        s_nPoints[threadIdx.x] = 0;
        s_sumX[threadIdx.x] = 0.0f;
        s_sumY[threadIdx.x] = 0.0f;
        s_centroids[2*threadIdx.x] = d_centroids[2*threadIdx.x];
        s_centroids[2*threadIdx.x+1] = d_centroids[2*threadIdx.x+1];
    }

    __syncthreads();

    if (idx < static_cast<int>(ceil(static_cast<float>(numPoints) / POINTS_PER_THREAD))) {
        for (int i = 0; i < POINTS_PER_THREAD; ++i) {
            int pointIdx = idx * POINTS_PER_THREAD + i;

            if (pointIdx < numPoints) {
                float minDist = FLT_MAX;
                int bestCluster = -1;
                float px = d_datapoints[2 * pointIdx];
                float py = d_datapoints[2 * pointIdx + 1];

                #pragma unroll 5
                for (int c = 0; c < k; ++c) {
                    float cx = s_centroids[2 * c];
                    float cy = s_centroids[2 * c + 1];
                    float dist = (px - cx) * (px - cx) + (py - cy) * (py - cy);

                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                d_clust_assn[pointIdx] = bestCluster;
                atomicAdd(&s_nPoints[bestCluster], 1);
                atomicAdd(&s_sumX[bestCluster], px);
                atomicAdd(&s_sumY[bestCluster], py);
            }
        }
    }

    __syncthreads();

   if (threadIdx.x < k) {
        atomicAdd(&d_nPoints[threadIdx.x], s_nPoints[threadIdx.x]);
        atomicAdd(&d_sumX[threadIdx.x], s_sumX[threadIdx.x]);
        atomicAdd(&d_sumY[threadIdx.x], s_sumY[threadIdx.x]);
    }

}

__global__ void updateCentroids(float *d_centroids, int *d_nPoints, float *d_sumX, float *d_sumY, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < static_cast<int>(ceil(static_cast<float>(k) / CLUSTERS_PER_THREAD))) {
        for (int i = 0; i < CLUSTERS_PER_THREAD; ++i) {
            int centroidIdx = idx * CLUSTERS_PER_THREAD + i;

            if (centroidIdx < k) {
                if (d_nPoints[centroidIdx] > 0) {
                    d_centroids[2 * centroidIdx] = d_sumX[centroidIdx] / d_nPoints[centroidIdx];
                    d_centroids[2 * centroidIdx + 1] = d_sumY[centroidIdx] / d_nPoints[centroidIdx];
                }
            }
        }
    }
}

std::vector<double> kmeans_cuda(std::vector<Point> &points, const std::vector<Point> &centroids, int k, int epochs, int numThreads, int numBlocksA, int numBlocksB) {
    int numPoints = points.size();
    int *clust_assn = new int[numPoints];

    float *d_datapoints;
    int *d_clust_assn;
    float *d_centroids;
    int *d_nPoints;
    float *d_sumX;
    float *d_sumY;

    cudaMalloc(&d_datapoints, numPoints * 2 * sizeof(float));
    cudaMalloc(&d_clust_assn, numPoints * sizeof(int));
    cudaMalloc(&d_centroids, k * 2 * sizeof(float));
    cudaMalloc(&d_nPoints, k * sizeof(int));
    cudaMalloc(&d_sumX, k * sizeof(float));
    cudaMalloc(&d_sumY, k * sizeof(float));

    auto pointsSoA = Points(numPoints, points);
    cudaMemcpy(d_datapoints, pointsSoA.flat_coord.data(), numPoints * 2 * sizeof(float), cudaMemcpyHostToDevice);

    auto centroidsSoA = Points(k, centroids);
    cudaMemcpy(d_centroids, centroidsSoA.flat_coord.data(), k * 2 * sizeof(float), cudaMemcpyHostToDevice);

    size_t sharedMemSize = (4 * k * sizeof(float))+(k*sizeof(int));
    double clusterAssignTime = 0.0;
    double centroidsUpdateTime = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        cudaMemset(d_nPoints, 0, k * sizeof(int));
        cudaMemset(d_sumX, 0, k * sizeof(float));
        cudaMemset(d_sumY, 0, k * sizeof(float));

        double startClusterAssign = omp_get_wtime();
        assignClusters<<<numBlocksA, numThreads, sharedMemSize>>>(d_datapoints, d_clust_assn, d_centroids, d_nPoints,d_sumX, d_sumY, numPoints, k);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Errore durante l'esecuzione del kernel: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        double endClusterAssign = omp_get_wtime();
        clusterAssignTime += (endClusterAssign - startClusterAssign);

        double startCentroidsUpdate = omp_get_wtime();
        updateCentroids<<<numBlocksB, numThreads>>>(d_centroids, d_nPoints, d_sumX, d_sumY, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Errore durante l'esecuzione del kernel: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        double endCentroidsUpdate = omp_get_wtime();
        centroidsUpdateTime += (endCentroidsUpdate - startCentroidsUpdate);
    }

    cudaMemcpy(pointsSoA.clusters.data(), d_clust_assn, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numPoints; ++i) {
        points[i].cluster = pointsSoA.clusters[i];
    }

    delete[] clust_assn;
    cudaFree(d_datapoints);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids);
    cudaFree(d_nPoints);
    cudaFree(d_sumX);
    cudaFree(d_sumY);

    return {clusterAssignTime, centroidsUpdateTime};
}

int main() {
    std::vector<int> numThreadsList = {1, 32, 64, 128, 256, 512, 1024};
    std::vector<int> numPointsList = {500, 7000, 20000, 50000, 126000};
    std::vector<std::tuple<int, int, double, double, double> > results, resultsAssignment, resultsCentroids;
    std::unordered_map<int, int> pointsToClusters = {{500, 5}, {7000, 10}, {20000, 12}, {50000, 15}, {126000, 18}, {301487, 20} };
    std::unordered_map<int, int> pointsToEpochs = {{500, 20}, {7000, 50}, {20000, 70}, {50000, 100}, {126000, 120}, {301487, 200}};

    float minX, minY, maxX, maxY;
    std::vector<Point> points = KMeansHelper::readAndNormalizeCSV("resources/a.csv", minX, minY, maxX, maxY);
    numPointsList.push_back(points.size());

    for (int numPoints: numPointsList) {
        int numClusters = pointsToClusters[numPoints];
        std::cout << "Number of points: " << numPoints << " -> Number of clusters: " << numClusters << std::endl;
        double avgDurationSequential = 0.0;
        std::vector<Point> centroids;
        KMeansHelper::init_centroids(numClusters, points, centroids);

        for (int numThreads: numThreadsList) {
            int numMeasurements = 5;
            double totalDuration = 0.0;
            double totalClusterAssignDuration = 0.0;
            double totalCentroidUpdateDuration = 0.0;
            float totalThreads = static_cast<float>(numPoints) / POINTS_PER_THREAD;
            int numBlocks = static_cast<int>(ceil(totalThreads / numThreads));
            float totalThreadsB = ceil(static_cast<float>(numClusters) / CLUSTERS_PER_THREAD);
            int numBlocksB = static_cast<int>(ceil(totalThreadsB / numThreads));

            for (int measurement = 0; measurement < numMeasurements; ++measurement) {
                cudaDeviceReset();
                std::vector<Point> pointsCopy(points.begin(), points.begin() + numPoints);
                std::vector<Point> centroidsCopy = centroids;

                if (measurement == 0) {
                    KMeansHelper::draw_chart_gnu(pointsCopy, "initial_points_number_" + std::to_string(numPoints) + ".png", minX, minY, maxX, maxY, numClusters);
                }

                std::vector<double> times;
                double startParallel = omp_get_wtime();
                if (numThreads == 1) {
                    times = KMeansHelper::kmeans_sequential(pointsCopy, centroidsCopy, numClusters,pointsToEpochs[numPoints]);
                } else {
                    times = kmeans_cuda(pointsCopy, centroidsCopy, numClusters, pointsToEpochs[numPoints], numThreads, numBlocks, numBlocksB);
                }
                double endParallel = omp_get_wtime();
                totalDuration += (endParallel - startParallel);
                totalClusterAssignDuration += times[0];
                totalCentroidUpdateDuration += times[1];

                if (measurement == 0) {
                    KMeansHelper::draw_chart_gnu(
                        pointsCopy,
                        "kmeans_numThreads_" + std::to_string(numThreads) + "_numPoints_" + std::to_string(numPoints) +
                        "_measurement_n_" + std::to_string(measurement) + ".png", minX, minY, maxX, maxY, numClusters);
                }
            }

            if (numThreads == 1) {
                avgDurationSequential = totalDuration / numMeasurements;
            }

            double avgDurationParallel = totalDuration / numMeasurements;
            double speedup = avgDurationSequential / avgDurationParallel;
            double efficiency = numThreads == 1 ? 1 : speedup / (static_cast<double>(numBlocks * numThreads) + static_cast<double>(numBlocksB * numThreads));

            double avgClusterAssignDuration = totalClusterAssignDuration / numMeasurements;
            double avgCentroidUpdateDuration = totalCentroidUpdateDuration / numMeasurements;

            double clusterAssignSpeedup = numThreads == 1 ? 1 :(avgDurationSequential / numMeasurements) / avgClusterAssignDuration;
            double centroidUpdateSpeedup = numThreads == 1 ? 1 :(avgDurationSequential / numMeasurements) / avgCentroidUpdateDuration;

            double clusterAssignEfficiency = numThreads == 1 ? 1 :clusterAssignSpeedup / (numBlocks * numThreads);
            double centroidUpdateEfficiency = numThreads == 1 ? 1 :centroidUpdateSpeedup / (numBlocksB * numThreads);

            std::cout << "Points: " << numPoints << ", Threads per block: " << numThreads <<
                    ", Avg Duration (Parallel): " << avgDurationParallel << "s, Speedup: " << speedup <<
                    ", Efficiency: " << efficiency << std::endl;

            results.emplace_back(numPoints, numThreads, avgDurationParallel, speedup, efficiency);
            resultsAssignment.emplace_back(numPoints, numThreads, avgClusterAssignDuration, clusterAssignSpeedup, clusterAssignEfficiency);
            resultsCentroids.emplace_back(numPoints, numThreads, avgCentroidUpdateDuration, centroidUpdateSpeedup, centroidUpdateEfficiency);
        }
    }

    KMeansHelper::logExecutionDetails("performance_log.csv", results);
    KMeansHelper::logExecutionDetails("performance_assignment_log.csv", resultsAssignment);
    KMeansHelper::logExecutionDetails("performance_centroids_log.csv", resultsCentroids);

    return 0;
}