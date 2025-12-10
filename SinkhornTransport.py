import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, RelaxedOneHotCategorical
import numpy as np
import time
import ot
from numpy.typing import ArrayLike

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SinkhornTransportModel:
    """
    Implements a optimal transport plan following the method presented in "Sinkhorn Distances: 
    Lightspeed computation of optimal transportation distances" by Cuturi, 2015.
    Computes the transportation matrix with the Sinkhorn-Knopp-Algorithm.
    For distributions a = u_s, b = u_t, we aim to find P with P1 = a and P^T=b for which 
    argmin_{P >= 0} <P,C> - epsilon h(P) to compute u_(s|t)
    """
    C: torch.Tensor

    def __init__(self, vocab, lam=1e-2, threshold=1e-7, C=None):
        # Default: C_{u,v} = 1_{u\neq v}
        if C is not None:
            self. C = C
        else:
            self.C: torch.Tensor = torch.ones((vocab, vocab)).fill_diagonal_(0)
        self.lam = lam
        self.threshold = threshold

    def compute_optimal_transport(self, a: torch.Tensor, b: torch.Tensor):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm with self.C (of shape (K, K) ) as cost matrix

        Args:
          a: source distribution, of shape (..., K)
          b: target distribution, of shape (..., K)
        Returns:
          P: optimal transport matrix of shape (K, K)
          cost: Minimal cost
        """
        ndim = a.ndim
        eps = 1e-8
        print(eps)
        P: torch.Tensor = torch.exp(-self.C / self.lam)
        P = P.view([1 for _ in range(ndim-1)]+[vocab, vocab])
        pdim = list(a.shape[:-1])+[1, 1]
        P = P.repeat(pdim)
        # make sure sum over last two dimension add up to 0
        P /= P.sum(dim=(-2, -1), keepdim=True) + eps

        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        b = b.reshape([*a.shape[:-2]]+[1, -1])  # to col vec

        err = 1
        i = 0
        while err > self.threshold:
            # different computation than in proposal and paper
            # but equivalent calculation
            row_ratio = a / (P.sum(dim=-1, keepdim=True) + eps)
            P *= row_ratio
            col_ratio = b / (P.sum(dim=-2, keepdim=True) + eps)
            P *= col_ratio

            err_row = torch.max(torch.abs(P.sum(dim=-1, keepdim=True) - a))
            err_col = torch.max(torch.abs(P.sum(dim=-2, keepdim=True) - b))

            err = torch.max(err_row, err_col)

            i += 1

            if i == 1500:
                print(
                    "Warning: Sinkhorn Transport Model did not converge after 1500 iterations")
                break
        # Sanity check
        # a = a.squeeze(-1)
        # b = b.squeeze(-2)
        # print(torch.allclose(P.sum(dim=(-1)), a, rtol=1e-05, atol=1e-08))
        # print(torch.allclose(P.sum(dim=(-2)), b, rtol=1e-05, atol=1e-08))
        return P, err

    def compute_posterior(self, u_s, u_t):
        '''
        Docstring for compute_posterior
        
        :param u_s: Marginal parameters at time s [batch, H, W, vocab]
        :param u_t: Marginal parameters at time t [batch, H, W, vocab]
        
        :out res: [batch, H, W, vocab, vocab]
        '''
        eps = 1e-8
        a, b = u_s, u_t
        P, err = self.compute_optimal_transport(
            a, b)  # shape [batch, H, W, vocab, vocab]
        # res [r, c] is equal to q(z_s = r | z_t = c)
        res = P / (P.sum(-2, keepdim=True) + eps)
        return res


if __name__ == "__main__":
    vocab = 3
    a = torch.Tensor([0.2, 0.2, 0.6])
    a = torch.rand(size=(16, 10, 10, vocab))
    a /= a.sum(dim=-1, keepdim=True)

    b = torch.Tensor([0.1, 0.8, 0.1])
    b = torch.rand(size=(16, 10, 10, vocab))
    b /= b.sum(dim=-1, keepdim=True)

    reg = 1
    start = time.time()
    tmodel = SinkhornTransportModel(vocab=vocab, lam=reg)
    print("before")
    transport_matrix, err = tmodel.compute_optimal_transport(
        a, b)

    pos = tmodel.compute_posterior(a, b)
    end = time.time()
    elapsedtime = end-start
    print(f"Own Time: {elapsedtime} s")
    print("Own error:", err)
    exit()

    # start = time.time()
    # gamma : ArrayLike = ot.sinkhorn(a, b, C, reg, stopThr=10e-8)  # type: ignore
    # end = time.time()
    # elapsedtime = end-start

    # print(f"POT time: {elapsedtime} s")
    # err_row = torch.max(torch.abs(gamma.sum(dim=-1) - a))
    # err_col = torch.max(torch.abs(gamma.sum(dim=-2) - b))
    # err = torch.max(err_row, err_col)
    # print("POT error:", err)

    print(transport_matrix)
    # print(gamma)
