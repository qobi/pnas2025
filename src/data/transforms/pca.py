from src.data.transforms.transform import Transform
from src.data.transforms.standardize import Standardize
from torch.linalg import svd
from functools import partial

class PCA(Transform):
    def __init__(self, scaler=Standardize()):
        self.scaler = scaler
        self.V = None

    def fit(self, data):
        if self.scaler:
            data = self.scaler.fit_transform(data)
        _, _, self.V = svd(data.reshape(data.shape[0], -1), full_matrices=False)

    def transform(self, data, n_components=None):
        if self.scaler:
            data = self.scaler.transform(data)
        if n_components:
            return data.reshape(data.shape[0], -1) @ self.V[:n_components].T
        else:
            return partial(self.transform, data)