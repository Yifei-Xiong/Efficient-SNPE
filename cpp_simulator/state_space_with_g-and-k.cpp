#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <ctime>
#include <thread>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

double compute_Mk(const std::vector<double>& samples, int k) {
    int n = samples.size();
    double Mk = 0.0;
    if (k == 0) {
        Mk = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
    } else {
        for (int i = 1; i <= n; ++i) {
            double weight = 1.0;
            for (int j = 0; j < k; ++j) {
                weight *= (i - 1 - j) / static_cast<double>(n - 1 - j);
            }
            Mk += weight * samples[i - 1];
        }
        Mk /= n;
    }

    return Mk;
}

std::vector<double> compute_L_moments(const std::vector<double>& samples) {
    std::vector<double> sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    double M0 = compute_Mk(sorted_samples, 0);
    double M1 = compute_Mk(sorted_samples, 1);
    double M2 = compute_Mk(sorted_samples, 2);
    double M3 = compute_Mk(sorted_samples, 3);

    std::vector<double> L_moments(4);
    L_moments[0] = M0;                                    // L1
    L_moments[1] = 2 * M1 - M0;                           // L2
    L_moments[2] = (6 * M2 - 6 * M1 + M0) / L_moments[1]; // L3
    L_moments[3] = (20 * M3 - 30 * M2 + 12 * M1 - M0) / L_moments[1]; // L4

    L_moments[1] = std::log(L_moments[1]);
    L_moments[3] = std::log(L_moments[3] + 0.5);

    return L_moments;
}

void gandk_main(const vector<double> &para, int seed, int N, double *result) {
    std::mt19937 gen(seed);

    // create transformed parameters
    double paraA = para[0];
    double paraB = std::exp(para[1]);
    double paraG = para[2];
    double paraK = std::exp(para[3]) - 0.5;

    // generate N standard normal random variables
    vector<double> z(N);
    std::normal_distribution<double> dnorm(0, 1);
    for (int i = 0; i < N; ++i) {
        z[i] = dnorm(gen);
    }

    std::vector<double> samples(N);
    for (int i = 0; i < N; ++i) {
        double term1 = (1 - std::exp(-paraG * z[i])) / (1 + std::exp(-paraG * z[i]));
        double term2 = std::pow(1 + z[i] * z[i], paraK);
        samples[i] = paraA + paraB * (1 + 0.8 * term1) * term2 * z[i];
    }

    auto L_moments = compute_L_moments(samples);

    result[0] = L_moments[0];
    result[1] = L_moments[1];
    result[2] = L_moments[2];
    result[3] = L_moments[3];
}

void sub_task(std::vector<double> &arr, int n_days) {
    const int T = n_days;
    // double ret[6 * 365];
    double *ret = new double[6 * n_days];
    for (int i = 0; i < T; ++i) {
        int start_pos = i * 6;
        vector<double> para = {arr[start_pos], arr[start_pos + 1], arr[start_pos + 2], arr[start_pos + 3]};
        int seed = int(arr[start_pos + 4]);
        int N = int(arr[start_pos + 5]);
        gandk_main(para, seed, N, ret + start_pos);
    }
    for (int i = 0; i < 6 * n_days; ++i) {
        arr[i] = ret[i];
    }
    delete[] ret;
}

#ifdef WIN32
#define EXTERN extern "C" __declspec(dllexport)
#elif __GNUC__
#define EXTERN extern "C"
#endif

EXTERN void gandk_multi_thread(double x[]) {
    int s = int(x[0]); // number of threads
    int k = static_cast<int>(std::round(x[1]));
    int n_days = int(x[2]);
    const int n = 6 * n_days; // length for each task
    std::vector<double> input(n * k); // storage all input
    std::vector<std::vector<double>> subArrays(k); // storage all input for each task
    for (int i = 0; i < n * k; ++i) {
        input[i] = x[i + 3];
    }
    for (int i = 0; i < k; ++i) {
        subArrays[i] = std::vector<double>(input.begin() + i * n, input.begin() + (i + 1) * n);
    }
    int subArraysPerThread = k / s;
    int remainingSubArrays = k % s;
    std::vector<std::thread> threads;
    int startIdx = 0;
    for (int i = 0; i < s; ++i) {
        int threadStartIdx = startIdx;
        int threadEndIdx = startIdx + subArraysPerThread + (i < remainingSubArrays ? 1 : 0);
        threads.emplace_back([threadStartIdx, threadEndIdx, &subArrays, n_days]() {
            for (int j = threadStartIdx; j < threadEndIdx; ++j) {
                sub_task(subArrays[j], n_days);
            }
        });
        startIdx = threadEndIdx;
    }
    for (auto &thread: threads) {
        thread.join();
    }
    int x_idx = 3;
    for (const auto &subArray: subArrays) {
        for (const auto &value: subArray) {
            x[x_idx] = value;
            x_idx++;
        }
    }
}


void compute_data_summary(const string& csv_path) {
    // read csv file
    std::ifstream file(csv_path);
    std::string line;
    std::vector<double> price;
    std::vector<int> day;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');
        price.push_back(std::stod(token));
        std::getline(ss, token, ',');
        day.push_back(std::stoi(token));
    }

    std::vector<double> day_i_price;
    // create 365 * 4 array to store L-moments
    auto L_moments = new double[365 * 4];
    auto n_i = new int[365];
    for (int i = 0; i < 365; ++i) {
        // add price to day_i_price if day == i
        day_i_price.clear();
        for (int j = 0; j < price.size(); ++j) {
            if (day[j] == i) {
                day_i_price.push_back(price[j]);
            }
        }
        n_i[i] = day_i_price.size();
        auto L_moments_i = compute_L_moments(day_i_price);
        for (int j = 0; j < 4; ++j) {
            L_moments[i * 4 + j] = L_moments_i[j];
        }
    }

    // save n_i and L-moments to csv
    std::ofstream output_file("data_summary.csv");
    output_file << "day,n_i,L1,L2,L3,L4\n";
    for (int i = 0; i < 365; ++i) {
        output_file << i << "," << n_i[i] << ",";
        for (int j = 0; j < 4; ++j) {
            output_file << L_moments[i * 4 + j] << ",";
        }
        output_file << "\n";
    }
}
