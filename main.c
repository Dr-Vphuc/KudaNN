#include <stdio.h>
#include <stdlib.h>
#include "kudann.h"

int main(void) {
    /* Dữ liệu giả lập để test nhanh */
    const int n_train = 2000;
    const int n_test  = 500;
    const int n_feat  = 16;
    const int k = 3;

    float *X_train = (float *)malloc((size_t)n_train * n_feat * sizeof(float));
    int   *y_train = (int *)malloc((size_t)n_train * sizeof(int));
    float *X_test  = (float *)malloc((size_t)n_test * n_feat * sizeof(float));
    int   *y_test  = (int *)malloc((size_t)n_test * sizeof(int));

    srand(42);
    for (int i = 0; i < n_train; ++i) {
        int cls = i % 3;
        y_train[i] = cls;
        for (int f = 0; f < n_feat; ++f) {
            X_train[i * n_feat + f] = (float)(cls * 5) + ((float)rand() / RAND_MAX);
        }
    }
    for (int i = 0; i < n_test; ++i) {
        int cls = i % 3;
        y_test[i] = cls;
        for (int f = 0; f < n_feat; ++f) {
            X_test[i * n_feat + f] = (float)(cls * 5) + ((float)rand() / RAND_MAX);
        }
    }

    int *preds_cpu  = (int *)malloc((size_t)n_test * sizeof(int));
    int *preds_cuda = (int *)malloc((size_t)n_test * sizeof(int));

    KudaNNClassifier clf_cpu, clf_cuda;
    kudann_init(&clf_cpu, k, MODE_CPU, 256);
    kudann_init(&clf_cuda, k, MODE_CUDA, 256);

    kudann_fit(&clf_cpu, X_train, y_train, n_train, n_feat);
    kudann_fit(&clf_cuda, X_train, y_train, n_train, n_feat);

    /* Warm-up GPU */
    kudann_predict(&clf_cuda, X_test, n_test, preds_cuda);

    double t_cpu  = kudann_predict_with_timing(&clf_cpu, X_test, n_test, preds_cpu);
    double t_cuda = kudann_predict_with_timing(&clf_cuda, X_test, n_test, preds_cuda);

    printf("CPU  Accuracy: %.4f | Time: %.6f s\n", kudann_score(y_test, preds_cpu, n_test), t_cpu);
    printf("CUDA Accuracy: %.4f | Time: %.6f s\n", kudann_score(y_test, preds_cuda, n_test), t_cuda);

    kudann_free(&clf_cpu);
    kudann_free(&clf_cuda);
    free(X_train); free(y_train); free(X_test); free(y_test);
    free(preds_cpu); free(preds_cuda);

    return 0;
}