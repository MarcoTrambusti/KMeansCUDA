#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <random>
#include <limits>

struct Point {
    float x, y;
    int cluster;
    float minDist;
    Point() : x(0.0), y(0.0), cluster(-1), minDist(std::numeric_limits<float>::max()) {}
    Point(float x, float y) : x(x), y(y), cluster(-1), minDist(std::numeric_limits<float>::max()) {}

    [[nodiscard]] float distance(Point p) const {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

class KMeansHelper {
public:
    static void init_centroids(int k, const std::vector<Point>& points, std::vector<Point>& centroids);
    static void draw_chart_gnu(const std::vector<Point>& points, const std::string& filename, float minX, float minY, float maxX, float maxY, int numClusters);
    static std::vector<Point> readAndNormalizeCSV(const std::string& filename, float& minX, float& minY, float& maxX, float& maxY);
    static void logExecutionDetails(const std::string& filename, const std::vector<std::tuple<int, int, double, double, double>>& results);
    static void kmeans_sequential(std::vector<Point>& points, std::vector<Point>& centroids, int k, int epochs);
};

#endif // COMMON_H