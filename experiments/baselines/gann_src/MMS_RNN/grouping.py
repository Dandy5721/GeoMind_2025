import torch
from sklearn.cluster import SpectralClustering


class Grouping:
    def __init__(self, bandwidth=50, **kwargs):
        self.bandwidth = bandwidth
        self.kwargs = kwargs

    def fit(self, X):
        S, U = torch.linalg.eigh(X.detach())
        S = S.log().diag_embed()
        log_X = U @ S @ U.transpose(-2, -1)
        pair_dis = torch.norm(
            log_X.unsqueeze(-4) - log_X.unsqueeze(-3), p='fro', dim=(-2, -1)
        )
        pair_dis_square = pair_dis**2
        affinity = torch.exp(-0.5 * pair_dis_square / (self.bandwidth * self.bandwidth))
        clustering = SpectralClustering(affinity='precomputed', **self.kwargs)
        self.labels_ = clustering.fit(affinity).labels_
        return self
