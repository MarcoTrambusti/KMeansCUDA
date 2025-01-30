#include "common.h"

void KMeansHelper::init_centroids(int k, const std::vector<Point>& points, std::vector<Point>& centroids) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    for (int i = 0; i < k; ++i) {
        centroids.push_back(points[dis(gen)]);
    }
}

void KMeansHelper::draw_chart_gnu(std::vector<Point>& points, const std::string& filename, double minX, double minY, double maxX, double maxY, int numClusters) {
    std::ofstream outfile("data.txt");
    std::filesystem::create_directory("plots");

    for (auto point : points) {
        outfile << point.x * (maxX - minX) + minX << " " << point.y * (maxY - minY) + minY << " " << point.cluster << std::endl;
    }

    outfile.close();
    std::string gnuplot_command = "gnuplot -e \"set terminal png size 800,600; set output 'plots/" + filename + "'; set xlabel 'Age'; set ylabel 'Total amount spent'; set palette rgbformulae 22,13,-31; set cbrange [0:" + std::to_string(numClusters) + "]; plot 'data.txt' using 1:2:3 with points pt 7 palette notitle\"";
    system(gnuplot_command.c_str());
    remove("data.txt");
}

std::vector<Point> KMeansHelper::readAndNormalizeCSV(const std::string& filename, double& minX, double& minY, double& maxX, double& maxY) {
    std::cout << "Reading CSV file..." << std::endl;
    std::vector<Point> points;
    std::string line;
    std::ifstream file(filename);
    std::cout << (file.is_open() ? "File Opened" : "Could not open the file!") << std::endl;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string bit;
        double x, y;
        std::getline(lineStream, bit, ',');
        x = std::stod(bit);
        std::getline(lineStream, bit, ',');
        y = std::stof(bit);
        points.emplace_back(x, y);
    }

    minX = std::min_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.x < b.x; })->x;
    maxX = std::max_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.x < b.x; })->x;
    minY = std::min_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.y < b.y; })->y;
    maxY = std::max_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.y < b.y; })->y;

    for (auto& point : points) {
        point.x = (point.x - minX) / (maxX - minX);
        point.y = (point.y - minY) / (maxY - minY);
    }

    return points;
}

void KMeansHelper::logExecutionDetails(const std::string& filename, const std::vector<std::tuple<int, int, double, double, double>>& results) {
    std::ofstream out(filename);
    out << "points,Threads,Duration (s),Speedup,Efficiency\n";
    for (const auto& result : results) {
        out << std::get<0>(result) << ","
            << std::get<1>(result) << ","
            << std::fixed << std::setprecision(4) << std::get<2>(result) << ","
            << std::fixed << std::setprecision(2) << std::get<3>(result) << ","
            << std::fixed << std::setprecision(2) << std::get<4>(result) << "\n";
    }
    out.close();
}

void KMeansHelper::kmeans_sequential(std::vector<Point>& points, std::vector<Point>& centroids, int k, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> sumX(k, 0.0f), sumY(k, 0.0f);
        std::vector<int> nPoints(k, 0);
        // Assign points to clusters
        for (auto& point : points) {
            for (int i = 0; i < k; ++i) {
                double dist = point.distance(centroids[i]);
                if (dist < point.minDist) {
                    point.minDist = dist;
                    point.cluster = i;
                }
            }
            //append data to centroids
            int clusterId = point.cluster;
            nPoints[clusterId]++;
            sumX[clusterId] += point.x;
            sumY[clusterId] += point.y;
            // reset distance
            point.minDist = std::numeric_limits<double>::max();
        }

        // Update centroids
        for (int i = 0; i < k; ++i) {
            if (nPoints[i] > 0) {
                double newX = sumX[i] / nPoints[i];
                double newY = sumY[i] / nPoints[i];
                centroids[i].x = newX;
                centroids[i].y = newY;
            }
        }
    }
}