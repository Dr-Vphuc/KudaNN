"""
Cài đặt thuật toán KNN tuần tự (CPU) và baseline sử dụng scikit-learn.
Dựa trên thuật toán Algorithm 1 và Flowchart (Hình 1) trong tài liệu báo cáo.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class SequentialKNN:
  """Cài đặt thuật toán K-Nearest Neighbors tuần tự thuần Python/NumPy.

  Thực hiện đúng logic của Algorithm 1 trong báo cáo:
  - Tính khoảng cách Euclidean từ từng điểm test tới toàn bộ tập train.
  - Sắp xếp khoảng cách theo thứ tự tăng dần.
  - Lấy K láng giềng gần nhất và thực hiện bỏ phiếu đa số (majority vote).
  """

  def __init__(self, k=3):
    self.k = k
    self.X_train = None
    self.y_train = None

  def fit(self, X_train, y_train):
    self.X_train = np.array(X_train)
    self.y_train = np.array(y_train)
    return self

  def _predict_single(self, x):
    # 1. Tính khoảng cách Euclidean từ x đến tất cả các điểm trong tập train
    # Công thức: d = sqrt(sum((x_train - x)^2))
    distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    # 2. Sắp xếp khoảng cách theo thứ tự tăng dần và lấy chỉ số (indices)
    sorted_indices = np.argsort(distances)

    # 3. Lấy ra nhãn của K láng giềng gần nhất
    k_indices = sorted_indices[: self.k]
    k_nearest_labels = self.y_train[k_indices]

    # 4. Bỏ phiếu đa số (Majority Vote) để tìm lớp xuất hiện nhiều nhất
    vals, counts = np.unique(k_nearest_labels, return_counts=True)
    majority_class = vals[np.argmax(counts)]

    return majority_class

  def predict(self, X_test):
    X_test = np.array(X_test)
    predictions = [self._predict_single(x) for x in X_test]
    return np.array(predictions)


class SklearnKNNBaseline:
  """Sử dụng thư viện scikit-learn để làm chuẩn đối chiếu (baseline)

  Dùng để kiểm tra độ chính xác (precision ~ 1.0 trên tập Iris như mục III trong báo cáo).
  """

  def __init__(self, k=3):
    self.model = KNeighborsClassifier(n_neighbors=k)

  def fit(self, X_train, y_train):
    self.model.fit(X_train, y_train)
    return self

  def predict(self, X_test):
    return self.model.predict(X_test)