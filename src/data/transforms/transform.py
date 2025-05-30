class Transform:
    
    def fit(self, data):
        """Fit the transform to the data."""
        raise NotImplementedError("fit method not implemented.")
    
    def transform(self, data):
        """Transform the data."""
        raise NotImplementedError("transform method not implemented.")
    
    def fit_transform(self, data, **kwargs):
        self.fit(data)
        return self.transform(data, **kwargs)
    
class Identity(Transform):
    
    def fit(self, data):
        """Identity transform does not require fitting."""
        pass
    
    def transform(self, data):
        """Returns the data unchanged."""
        return data
    
class Flatten(Transform):
    def fit(self, data):
        """Flatten transform does not require fitting."""
        pass
    
    def transform(self, data):
        """Flattens the input data."""
        return data.view(data.size(0), -1)

class Unsqueeze(Transform):
    def __init__(self, dim):
        """Unsqueeze the data along the specified dimension."""
        self.dim = dim
    
    def fit(self, data):
        """Unsqueeze transform does not require fitting."""
        pass
    
    def transform(self, data):
        """Unsqueezes the input data along the specified dimension."""
        return data.unsqueeze(self.dim)  # Add a dimension at the specified index