from src.data.transforms.transform import Transform

class Standardize(Transform):
    def __init__(self, dim=None):
        self.dim = dim
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(self.dim, keepdim=True) if self.dim else data.mean()
        self.std = data.std(self.dim, keepdim=True) if self.dim else data.std()
    
    def transform(self, data):
        data = (data - self.mean) / (self.std + 1e-8)
        return data

class Normalize(Transform):
    def __init__(self):

        self.max = None
        self.min = None

    def fit(self, data):
        self.max = data.max()
        self.min = data.min()
    
    def transform(self, data):
        data = (data - self.min) / (self.max - self.min)
        return data