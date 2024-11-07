#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <ctime>
#include <thread>
#include <chrono>

using namespace std;

void lotka_main(const vector<double>& para,int seed, double* result) {
    const double dt = 0.2;
    const double total_time = 30.0;
    const int sim_step = static_cast<int>(total_time / dt) + 1; // 151
    int state[2] = {50, 100};
    double current_time = 0.0;
    double event_time = 0.0;
    int iter_num = 0;
    int total_state[sim_step][2];
    srand(time(NULL));
    std::random_device rd;
    std::mt19937 gen(rand() + seed + rd());
    std::uniform_real_distribution<> sampler(0.0, 1.0);
    for (int j = 0; j < sim_step; j++) {
        while (current_time > event_time) {
            vector<double> rate = {state[0] * state[1] * para[0], state[0] * para[1], state[1] * para[2],
                                   state[0] * state[1] * para[3]};
            double total_rate = accumulate(rate.begin(), rate.end(), 0.0);
            if (total_rate == 0 || iter_num >= 10000 || state[0] >= 1000 || state[1] >= 1000) {
                event_time = numeric_limits<double>::infinity();
                break;
            }
            event_time += -log(sampler(gen)) / total_rate;
            double event_idx = sampler(gen);
            iter_num++;
            if (event_idx < rate[0] / total_rate) {
                state[0] += 1;
            } else if (event_idx < (rate[0] + rate[1]) / total_rate) {
                state[0] -= 1;
            } else if (event_idx < (rate[0] + rate[1] + rate[2]) / total_rate) {
                state[1] += 1;
            } else {
                state[1] -= 1;
            }
        }
        total_state[j][0] = state[0];
        total_state[j][1] = state[1];
        current_time += dt;
    }
    double sumx = 0.0;
    double sumy = 0.0;
    for (auto i = 0; i < sim_step; i++) {
        sumx += total_state[i][0];
        sumy += total_state[i][1];
    }
    double meanx = sumx / sim_step;
    double meany = sumy / sim_step;
    double varx = 0.0;
    double vary = 0.0;
    for (auto i = 0; i < sim_step; i++) {
        varx += pow(total_state[i][0] - meanx, 2);
        vary += pow(total_state[i][1] - meany, 2);
    }
    varx = varx / sim_step + 1e-4;
    vary = vary / sim_step + 1e-4;
    double logvarx = log(varx + 1);
    double logvary = log(vary + 1);

    double total_state_sta[sim_step][2];
    for (size_t i = 0; i < sim_step; i++) {
        total_state_sta[i][0] = (total_state[i][0] - meanx) / sqrt(varx);
        total_state_sta[i][1] = (total_state[i][1] - meany) / sqrt(vary);
    }

    double acx1 = 0.0;
    for (size_t i = 0; i < (sim_step-1); i++) {
        acx1 += total_state_sta[i][0] * total_state_sta[i+1][0];
    }
    acx1 = acx1 / (sim_step-1);

    double acx2 = 0.0;
    for (size_t i = 0; i < (sim_step-2); i++) {
        acx2 += total_state_sta[i][0] * total_state_sta[i+2][0];
    }
    acx2 = acx2 / (sim_step-1);

    double acy1 = 0.0;
    for (size_t i = 0; i < (sim_step-1); i++) {
        acy1 += total_state_sta[i][1] * total_state_sta[i+1][1];
    }
    acy1 = acy1 / (sim_step-1);

    double acy2 = 0.0;
    for (size_t i = 0; i < (sim_step-2); i++) {
        acy2 += total_state_sta[i][1] * total_state_sta[i+2][1];
    }
    acy2 = acy2 / (sim_step-1);

    double ccxy = 0.0;
    for (size_t i = 0; i < sim_step; i++) {
        ccxy += total_state_sta[i][0] * total_state_sta[i][1];
    }
    ccxy = ccxy / (sim_step-1);

    result[0] = meanx;
    result[1] = meany;
    result[2] = logvarx;
    result[3] = logvary;
    result[4] = acx1;
    result[5] = acx2;
    result[6] = acy1;
    result[7] = acy2;
    result[8] = ccxy;
}

// subtask for multi thread, generate one observation
void lotka_sub_task(std::vector<double> &arr) {
    vector<double> para = {arr[0], arr[1], arr[2], arr[3]};
    int seed = int(arr[4]);
    double ret[9];
    lotka_main(para, seed, ret);
    for (int i = 0; i < 9; ++i) {
        arr[i] = ret[i];
    }
}

#ifdef WIN32
#define EXTERN extern "C" __declspec(dllexport)
#elif __GNUC__
#define EXTERN extern "C"
#endif

EXTERN void lotka_multi_thread(double x[]) {
    const int n = 9; // length for each task
    const int s = int(x[0]); // number of threads
    const int k = int(x[1]); // number of tasks
    std::vector<double> input(n * k); // storage all input
    std::vector<std::vector<double>> subArrays(k); // storage all input for each task
    for (int i = 0; i < n * k; ++i) {
        input[i] = x[i + 2];
    }
    for (int i = 0; i < k; ++i) {
        subArrays[i] = std::vector<double>(input.begin() + i * n, input.begin() + (i + 1) * n);
    }
    int subArraysPerThread = k / s;
    int remainingSubArrays = k % s;
    std::vector<std::thread> threads;
    int startIdx = 0;
    for (int i = 0; i < s; ++i) {
        int endIdx = startIdx + subArraysPerThread + (i < remainingSubArrays ? 1 : 0);
        threads.emplace_back([startIdx, endIdx, &subArrays]() {
            for (int j = startIdx; j < endIdx; ++j) {
                lotka_sub_task(subArrays[j]);
            }
        });
        startIdx = endIdx;
    }
    for (auto &thread: threads) {
        thread.join();
    }
    int x_idx = 2;
    for (const auto &subArray: subArrays) {
        for (const auto &value: subArray) {
            x[x_idx] = value;
            x_idx++;
        }
    }
}
