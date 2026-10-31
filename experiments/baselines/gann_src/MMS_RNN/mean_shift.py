import torch
from torch import nn


class SPD_GBMS_RNN(nn.Module):
    def __init__(self, bandwidth=0.5):
        super(SPD_GBMS_RNN, self).__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth))

    def log(self, X):
        S, U = torch.linalg.eigh(X)
        S = S.log().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def exp(self, X):
        S, U = torch.linalg.eigh(X)
        S = S.exp().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def logm(self, X, Y):
        return self.log(Y) - self.log(X)

    def expm(self, X, Y):
        return self.exp(self.log(X) + Y)

    def forward(self, X):
        X = X.squeeze()
        bandwidth = self.bandwidth
        log_X = self.log(X)
        pair_dis = torch.norm(
            log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1)
        )
        log_Y_X = log_X.unsqueeze(-4) - log_X.unsqueeze(-3)
        pair_dis_square = pair_dis**2
        W = torch.exp(-0.5 * pair_dis_square / (bandwidth * bandwidth))
        D = W.sum(dim=-1).diag_embed()
        D_inv = D.inverse()
        M = (
            (log_Y_X.permute(2, 3, 0, 1) @ W).diagonal(dim1=-2, dim2=-1) @ D_inv
        ).permute(2, 0, 1)
        output = self.expm(X, M)

        return output


def distance_loss(inputs, targets):
    identity_matrix = targets.unsqueeze(0) == targets.unsqueeze(0).T
    S, U = torch.linalg.eigh(inputs)
    S = S.log().diag_embed()
    log_X = U @ S @ U.transpose(-2, -1)
    pair_dis = torch.norm(
        log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-4, p='fro', dim=(-2, -1)
    )

    loss = pair_dis.cpu() * identity_matrix.cpu() - pair_dis.cpu() * (
        ~identity_matrix.cpu()
    )
    loss = loss.mean()
    return loss
