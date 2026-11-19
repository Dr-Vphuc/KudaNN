#ifndef KUDANN_H
#define KUDANN_H

#include <stddef.h>

typedef enum {
    MODE_CPU = 0,
    MODE_CUDA = 1
} KudaNNMode;

typedef struct {
    int k;
    KudaNNMode mode;
    int threads_per_block;

    float *X_train;   /* Kích thước: n_train * n_features */
    int *y_train;     /* Kích thước: n_train */
    int n_train;
    int n_features;
} KudaNNClassifier;

/* Khởi tạo & giải phóng */
void kudann_init(KudaNNClassifier *clf, int k, KudaNNMode mode, int threads_per_block);
void kudann_fit(KudaNNClassifier *clf, const float *X_train, const int *y_train, int n_train, int n_features);
void kudann_free(KudaNNClassifier *clf);

/* Dự đoán & đánh giá */
void kudann_predict(const KudaNNClassifier *clf, const float *X_test, int n_test, int *preds);
double kudann_predict_with_timing(const KudaNNClassifier *clf, const float *X_test, int n_test, int *preds);
double kudann_score(const int *y_true, const int *y_pred, int n_samples);

/* Hàm cài đặt lõi CPU & CUDA */
void knn_cpu_predict(
    const float *X_train, const int *y_train, int n_train,
    const float *X_test, int n_test, int n_features,
    int k, int *preds
);

void knn_cuda_predict(
    const float *X_train, const int *y_train, int n_train,
    const float *X_test, int n_test, int n_features,
    int k, int threads_per_block, int *preds
);

#endif