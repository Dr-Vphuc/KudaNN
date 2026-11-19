#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kudann.h"

void kudann_init(KudaNNClassifier *clf, int k, KudaNNMode mode, int threads_per_block) {
    clf->k = k;
    clf->mode = mode;
    clf->threads_per_block = (threads_per_block > 0) ? threads_per_block : 256;
    clf->X_train = NULL;
    clf->y_train = NULL;
    clf->n_train = 0;
    clf->n_features = 0;
}

void kudann_fit(KudaNNClassifier *clf, const float *X_train, const int *y_train, int n_train, int n_features) {
    free(clf->X_train);
    free(clf->y_train);

    clf->n_train = n_train;
    clf->n_features = n_features;

    size_t x_bytes = (size_t)n_train * n_features * sizeof(float);
    size_t y_bytes = (size_t)n_train * sizeof(int);

    clf->X_train = (float *)malloc(x_bytes);
    clf->y_train = (int *)malloc(y_bytes);

    memcpy(clf->X_train, X_train, x_bytes);
    memcpy(clf->y_train, y_train, y_bytes);
}

void kudann_free(KudaNNClassifier *clf) {
    free(clf->X_train);
    free(clf->y_train);
    clf->X_train = NULL;
    clf->y_train = NULL;
}

void kudann_predict(const KudaNNClassifier *clf, const float *X_test, int n_test, int *preds) {
    if (clf->mode == MODE_CUDA) {
        knn_cuda_predict(
            clf->X_train, clf->y_train, clf->n_train,
            X_test, n_test, clf->n_features,
            clf->k, clf->threads_per_block, preds
        );
    } else {
        knn_cpu_predict(
            clf->X_train, clf->y_train, clf->n_train,
            X_test, n_test, clf->n_features,
            clf->k, preds
        );
    }
}

double kudann_predict_with_timing(const KudaNNClassifier *clf, const float *X_test, int n_test, int *preds) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    kudann_predict(clf, X_test, n_test, preds);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (double)(end.tv_sec - start.tv_sec) +
                     (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double kudann_score(const int *y_true, const int *y_pred, int n_samples) {
    if (n_samples <= 0) return 0.0;
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return (double)correct / (double)n_samples;
}