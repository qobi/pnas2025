from src.data.transforms.transform import Transform
from src.data.feature_engineering import process_electrode_distance_matrix
import torch

class COO(Transform):
    def __init__(self, tau_dist=0.2, tau_fc=0.8, attrs = []):
        self.tau_dist = tau_dist
        self.tau_fc = tau_fc
        self.dist_conn = (process_electrode_distance_matrix() <= tau_dist).unsqueeze(0)
        n_electrodes = self.dist_conn.shape[1]
        self.self_conn = torch.eye(n_electrodes, dtype=torch.bool)#.unsqueeze(0)
        
        self.attrs = attrs

    def fit(self, _):
        pass
    
    def transform(self, fc):
        fc_conn = fc >= self.tau_fc
        connectivity_matrix = torch.logical_or(self.dist_conn.to(fc_conn.device), fc_conn)
        # connectivity_matrix = torch.logical_and(connectivity_matrix, ~self.self_conn.to(fc_conn.device))
        connectivity_matrix[:, self.self_conn.to(fc_conn.device)] = False

        edge_idx = []
        edge_attr = []
        for i, edges  in enumerate(connectivity_matrix):
            idx = torch.nonzero(edges).T.contiguous()
            edge_idx.append(idx)
            if len(self.attrs) == 0:
                continue
            attrs = torch.stack([attr[i, idx[0], idx[1]] for attr in self.attrs], dim=0).T.contiguous()
            edge_attr.append(attrs)
        
        return edge_idx if len(self.attrs) == 0 else (edge_idx, edge_attr)