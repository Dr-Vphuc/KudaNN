"""
Lớp giao diện thống nhất (Unified Interface) cho thuật toán KudaNN.
Giúp các thành viên khác trong nhóm dễ dàng sử dụng, chuyển đổi qua lại giữa 
CPU tuần tự, GPU CUDA và scikit-learn baseline chỉ bằng một tham số `mode`.
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score

from .knn_cpu import SequentialKNN, SklearnKNNBaseline
from .knn_cuda import CudaKNN


class KudaNNClassifier:
    """
    Giao diện chuẩn hóa (tương thích chuẩn scikit-learn) cho toàn bộ nhóm.
    
    Tham số:
    ----------
    k : int, default=3
        Số lượng láng giềng gần nhất (K).
    mode : {'cuda', 'cpu', 'sklearn'}, default='cuda'
        Chế độ thực thi thuật toán:
        - 'cuda'   : Chạy song song trên GPU bằng nhân CUDA (KudaNN).
        - 'cpu'    : Chạy tuần tự trên CPU bằng Python/NumPy.
        - 'sklearn': Chạy bằng thư viện scikit-learn chuẩn để đối chiếu độ chính xác.
    threads_per_block : int, default=256
        Số lượng threads trên mỗi block khi chạy ở chế độ 'cuda' (mặc định 256 theo báo cáo).
    """

    def __init__(self, k=3, mode="cuda", threads_per_block=256):
        self.k = k
        self.mode = mode.lower()
        self.threads_per_block = threads_per_block

        if self.mode == "cuda":
            self._model = CudaKNN(k=self.k, threads_per_block=self.threads_per_block)
        elif self.mode == "cpu":
            self._model = SequentialKNN(k=self.k)
        elif self.mode == "sklearn":
            self._model = SklearnKNNBaseline(k=self.k)
        else:
            raise ValueError(
                f"Chế độ mode='{mode}' không hợp lệ. Vui lòng chọn 'cuda', 'cpu' hoặc 'sklearn'."
            )

    def fit(self, X_train, y_train):
        """
        Nạp dữ liệu huấn luyện vào mô hình.
        """
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """
        Dự đoán nhãn cho tập dữ liệu kiểm thử X_test.
        """
        return self._model.predict(X_test)

    def predict_with_timing(self, X_test):
        """
        Dự đoán nhãn đồng thời đo chính xác thời gian thực thi (tính bằng giây).
        Hữu ích cho các thành viên làm nhiệm vụ benchmark và vẽ biểu đồ báo cáo.
        
        Trả về:
        -------
        predictions : np.ndarray
            Mảng nhãn dự đoán.
        elapsed_time : float
            Thời gian thực thi tính bằng giây.
        """
        start_time = time.perf_counter()
        predictions = self.predict(X_test)
        elapsed_time = time.perf_counter() - start_time
        return predictions, elapsed_time

    def score(self, X_test, y_test):
        """
        Tính độ chính xác (Accuracy / Precision) trên tập kiểm thử.
        """
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)
