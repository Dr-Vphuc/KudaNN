#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "kudann.h"
}

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

__global__ void knn_cuda_kernel(
    const float * __restrict__ X_train,
    const int * __restrict__ y_train,
    int n_train,
    const float * __restrict__ X_test,
    int n_test,
    int n_features,
    int k,
    int * __restrict__ predictions,
    float * __restrict__ dist_buffer,
    int * __restrict__ label_buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test) return;

    float *d_row = dist_buffer + (size_t)idx * n_train;
    int *l_row = label_buffer + (size_t)idx * n_train;
    const float *q_row = X_test + (size_t)idx * n_features;

    /* 1. Tính khoảng cách Euclidean tới toàn bộ tập train */
    for (int i = 0; i < n_train; ++i) {
        const float *tr_row = X_train + (size_t)i * n_features;
        float dist = 0.0f;
        for (int f = 0; f < n_features; ++f) {
            float diff = tr_row[f] - q_row[f];
            dist += diff * diff;
        }
        d_row[i] = sqrtf(dist);
        l_row[i] = y_train[i];
    }

    /* 2. Selection sort lấy K phần tử nhỏ nhất */
    for (int i = 0; i < k; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < n_train; ++j) {
            if (d_row[j] < d_row[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            float tmp_d = d_row[i];
            d_row[i] = d_row[min_idx];
            d_row[min_idx] = tmp_d;

            int tmp_l = l_row[i];
            l_row[i] = l_row[min_idx];
            l_row[min_idx] = tmp_l;
        }
    }

    /* 3. Majority vote */
    int best_label = l_row[0];
    int best_count = 0;
    for (int i = 0; i < k; ++i) {
        int cur = l_row[i];
        int cnt = 0;
        for (int j = 0; j < k; ++j) {
            if (l_row[j] == cur) cnt++;
        }
        if (cnt > best_count) {
            best_count = cnt;
            best_label = cur;
        }
    }

    predictions[idx] = best_label;
}

extern "C" void knn_cuda_predict(
    const float *X_train, const int *y_train, int n_train,
    const float *X_test, int n_test, int n_features,
    int k, int threads_per_block, int *preds
) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        fprintf(stderr, "Không tìm thấy GPU hỗ trợ CUDA, chuyển sang chạy CPU.\n");
        knn_cpu_predict(X_train, y_train, n_train, X_test, n_test, n_features, k, preds);
        return;
    }

    float *d_X_train = NULL, *d_X_test = NULL, *d_dist_buf = NULL;
    int *d_y_train = NULL, *d_preds = NULL, *d_label_buf = NULL;

    size_t train_x_bytes = (size_t)n_train * n_features * sizeof(float);
    size_t train_y_bytes = (size_t)n_train * sizeof(int);
    size_t test_x_bytes  = (size_t)n_test * n_features * sizeof(float);
    size_t preds_bytes   = (size_t)n_test * sizeof(int);
    size_t buf_dist_bytes  = (size_t)n_test * n_train * sizeof(float);
    size_t buf_label_bytes = (size_t)n_test * n_train * sizeof(int);

    CUDA_CHECK(cudaMalloc((void **)&d_X_train, train_x_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_y_train, train_y_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_X_test, test_x_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_preds, preds_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_dist_buf, buf_dist_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_label_buf, buf_label_bytes));

    CUDA_CHECK(cudaMemcpy(d_X_train, X_train, train_x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train, train_y_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X_test, X_test, test_x_bytes, cudaMemcpyHostToDevice));

    int blocks = (n_test + threads_per_block - 1) / threads_per_block;
    knn_cuda_kernel<<<blocks, threads_per_block>>>(
        d_X_train, d_y_train, n_train,
        d_X_test, n_test, n_features,
        k, d_preds, d_dist_buf, d_label_buf
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(preds, d_preds, preds_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_X_train);
    cudaFree(d_y_train);
    cudaFree(d_X_test);
    cudaFree(d_preds);
    cudaFree(d_dist_buf);
    cudaFree(d_label_buf);
}