"""
Cài đặt thuật toán KudaNN (KNN song song hóa trên GPU sử dụng CUDA).
Dựa trên mô hình PRAM (CREW) - Algorithm 2, Hình 2 và Listing 3 trong báo cáo.

Yêu cầu thư viện: numba, numpy
(Lưu ý: Dùng Numba CUDA cho phép viết trực tiếp nhân __global__ chuẩn C-CUDA 
ngay trong Python mà không cần cài đặt trình biên dịch nvcc phức tạp bên ngoài,
rất thuận tiện để các thành viên khác trong nhóm import và chạy thử trên Google Colab hoặc máy cá nhân).
"""

import math
import numpy as np
from numba import cuda


@cuda.jit
def knn_cuda_kernel(
    X_train,
    y_train,
    n_train,
    X_test,
    n_test,
    n_features,
    k,
    predictions,
    dist_buffer,
    label_buffer
):
    """
    Nhân CUDA thực hiện KNN song song cho từng điểm dữ liệu kiểm thử (test sample).
    Mỗi thread trên GPU chịu trách nhiệm dự đoán nhãn cho 1 điểm trong X_test.
    """
    # Xác định chỉ số của thread hiện tại (tương đương blockIdx.x * blockDim.x + threadIdx.x)
    idx = cuda.grid(1)

    if idx < n_test:
        # 1. Tính khoảng cách Euclidean từ điểm test[idx] tới tất cả các điểm trong X_train
        for i in range(n_train):
            dist = 0.0
            for f in range(n_features):
                diff = X_train[i, f] - X_test[idx, f]
                dist += diff * diff
            dist = math.sqrt(dist)

            dist_buffer[idx, i] = dist
            label_buffer[idx, i] = y_train[i]

        # 2. Sắp xếp mảng khoảng cách theo thứ tự tăng dần (Bubble Sort như Listing 3 trong báo cáo)
        # Có thể tối ưu bằng Selection Sort chỉ lấy K phần tử đầu để tăng tốc độ thực tế
        for i in range(k):
            min_idx = i
            for j in range(i + 1, n_train):
                if dist_buffer[idx, j] < dist_buffer[idx, min_idx]:
                    min_idx = j
            if min_idx != i:
                # Hoán đổi khoảng cách
                tmp_dist = dist_buffer[idx, i]
                dist_buffer[idx, i] = dist_buffer[idx, min_idx]
                dist_buffer[idx, min_idx] = tmp_dist
                # Hoán đổi nhãn tương ứng
                tmp_lbl = label_buffer[idx, i]
                label_buffer[idx, i] = label_buffer[idx, min_idx]
                label_buffer[idx, min_idx] = tmp_lbl

        # 3. Bỏ phiếu đa số (Majority Vote) trên K láng giềng gần nhất
        best_label = label_buffer[idx, 0]
        best_count = 0

        for i in range(k):
            current_label = label_buffer[idx, i]
            count = 0
            for j in range(k):
                if label_buffer[idx, j] == current_label:
                    count += 1
            if count > best_count:
                best_count = count
                best_label = current_label

        # Ghi kết quả dự đoán vào mảng đầu ra trên Global Memory
        predictions[idx] = best_label


class CudaKNN:
    """
    Lớp quản lý cấp phát bộ nhớ Host-Device và thực thi nhân CUDA cho KNN.
    """

    def __init__(self, k=3, threads_per_block=256):
        self.k = k
        self.threads_per_block = threads_per_block
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Đưa dữ liệu về kiểu float32 và int32 chuẩn bộ nhớ GPU
        self.X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        self.y_train = np.ascontiguousarray(y_train, dtype=np.int32)
        return self

    def predict(self, X_test):
        if not cuda.is_available():
            raise RuntimeError("Không tìm thấy GPU hỗ trợ CUDA trên thiết bị này!")

        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        n_train, n_features = self.X_train.shape
        n_test = X_test.shape[0]

        # Cấp phát bộ nhớ trên Device (GPU)
        d_X_train = cuda.to_device(self.X_train)
        d_y_train = cuda.to_device(self.y_train)
        d_X_test = cuda.to_device(X_test)
        d_predictions = cuda.device_array(n_test, dtype=np.int32)

        # Bộ đệm lưu khoảng cách và nhãn tạm cho từng thread trên GPU
        d_dist_buffer = cuda.device_array((n_test, n_train), dtype=np.float32)
        d_label_buffer = cuda.device_array((n_test, n_train), dtype=np.int32)

        # Tính số lượng blocks theo công thức (1) trong báo cáo:
        # N_blocks = (N_predictions + N_threads - 1) // N_threads
        blocks_per_grid = (n_test + self.threads_per_block - 1) // self.threads_per_block

        # Khởi chạy nhân CUDA
        knn_cuda_kernel[blocks_per_grid, self.threads_per_block](
            d_X_train,
            d_y_train,
            n_train,
            d_X_test,
            n_test,
            n_features,
            self.k,
            d_predictions,
            d_dist_buffer,
            d_label_buffer
        )

        # Đồng bộ hóa GPU và sao chép kết quả về Host (CPU)
        cuda.synchronize()
        return d_predictions.copy_to_host()
