#include <stdlib.h>
#include <math.h>
#include "kudann.h"

typedef struct {
    float dist;
    int label;
} Neighbor;

static int cmp_neighbor(const void *a, const void *b) {
    const Neighbor *na = (const Neighbor *)a;
    const Neighbor *nb = (const Neighbor *)b;
    if (na->dist < nb->dist) return -1;
    if (na->dist > nb->dist) return 1;
    return 0;
}

void knn_cpu_predict(
    const float *X_train, const int *y_train, int n_train,
    const float *X_test, int n_test, int n_features,
    int k, int *preds
) {
    Neighbor *neighbors = (Neighbor *)malloc((size_t)n_train * sizeof(Neighbor));
    if (!neighbors) return;

    for (int t = 0; t < n_test; ++t) {
        const float *x_q = X_test + (size_t)t * n_features;

        for (int i = 0; i < n_train; ++i) {
            const float *x_i = X_train + (size_t)i * n_features;
            float sum_sq = 0.0f;
            for (int f = 0; f < n_features; ++f) {
                float diff = x_i[f] - x_q[f];
                sum_sq += diff * diff;
            }
            neighbors[i].dist = sqrtf(sum_sq);
            neighbors[i].label = y_train[i];
        }

        qsort(neighbors, (size_t)n_train, sizeof(Neighbor), cmp_neighbor);

        /* Bỏ phiếu đa số trong K láng giềng gần nhất */
        int best_label = neighbors[0].label;
        int best_count = 0;

        for (int i = 0; i < k; ++i) {
            int current_label = neighbors[i].label;
            int count = 0;
            for (int j = 0; j < k; ++j) {
                if (neighbors[j].label == current_label) {
                    count++;
                }
            }
            if (count > best_count) {
                best_count = count;
                best_label = current_label;
            }
        }

        preds[t] = best_label;
    }

    free(neighbors);
}