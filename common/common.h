#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <filesystem>
#include <random>
#include <limits>
#include <iostream>

struct Point {
    double x, y;
    int cluster;
    double minDist;
    Point() : x(0.0), y(0.0), cluster(-1), minDist(std::numeric_limits<double>::max()) {}
    Point(double x, double y) : x(x), y(y), cluster(-1), minDist(std::numeric_limits<double>::max()) {}

    [[nodiscard]] double distance(Point p) const {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

class KMeansHelper {
public:
    static void init_centroids(int k, const std::vector<Point>& points, std::vector<Point>& centroids);
    static void draw_chart_gnu(std::vector<Point>& points, const std::string& filename, double minX, double minY, double maxX, double maxY, int numClusters);
    static std::vector<Point> readAndNormalizeCSV(const std::string& filename, double& minX, double& minY, double& maxX, double& maxY);
    static void logExecutionDetails(const std::string& filename, const std::vector<std::tuple<int, int, double, double, double>>& results);
    static void kmeans_sequential(std::vector<Point>& points, std::vector<Point>& centroids, int k, int epochs);
};

#endif // COMMON_H