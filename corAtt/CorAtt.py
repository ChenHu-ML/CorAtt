import torch
import torch.nn as nn
import corAtt.functionals as functions
from typing import Any, List, Tuple
from torch import Tensor

###  CorAtt layer
def cayley_map(X: torch.Tensor) -> torch.Tensor:
    """
    Formula:
        C(X) = (I_{n}-X)*(I_{n}+X)^{-1}
    """
    Id = torch.eye(X.size(-1), dtype=X.dtype, device=X.device)
    return (Id - X) @ torch.inverse(Id + X)


def matrix2skew(x: torch.Tensor) -> torch:
    return x - x.transpose(-1, -2)


def hol_plus_finder(x, atol=1e-3, max_iter=100):
    """
    Finding the unique diagonal matrix D such that exp(hol_mat + D).
    """
    d = torch.zeros_like(x, dtype=x.dtype) # setting d0

    for i in range(max_iter): # update rule: D_{k+1} = D_k - log(diag(exp(H + D_k)))
        d_next = d - functions.sym_expm.apply(d + x).diagonal(dim1=-2, dim2=-1).log.diag_embed()

        if torch.norm(d_next - d, dim=(-2, -1)).max() < atol: return d_next # Check convergence
        d = d_next

    return d


def spd_scaling_finder(spd_matrix, atol=1e-5, max_iter=100, damped=True):
    def func(A, x): return (A @ x.unsqueeze(-1) - 1 / x.unsqueeze(-1)).squeeze(-1) # f(x) = A @ x - 1/x
    def jac(A, x): return A + torch.diag_embed(1 / (x ** 2))  # Jacobian = A + diag(1/x^2)
    x = torch.ones(spd_matrix.shape[:-1], dtype=spd_matrix.dtype, device=spd_matrix.device)

    for _ in range(max_iter):  # damped Newton iterations
        f = func(spd_matrix, x)
    
        if torch.norm(f, dim=-1).max() < atol:
            break
        step = torch.linalg.solve(jac(spd_matrix, x), f)
        lam = torch.sqrt((f * step).sum(dim=-1, keepdim=True)) if damped else 0.0
        x -= step / (1 + lam)
    return x


class CorrAttention(nn.Module):
    def __init__(self, d_in, d_out, metric="olm", device='cpu', spe_device='cpu', dtype=torch.float):
        super(CorrAttention, self).__init__()
        self.device, self.dtype, self.metric  = device, dtype, metric
        self.d_in, self.d_out  = d_in, d_out

        self.q_weight = nn.Parameter((torch.randn(d_in, d_out)*2-1).to(self.device, self.dtype),requires_grad=True)
        self.k_weight = nn.Parameter((torch.randn(d_in, d_out)*2-1).to(self.device, self.dtype),requires_grad=True)
        self.v_weight = nn.Parameter((torch.randn(d_in, d_out)*2-1).to(self.device, self.dtype),requires_grad=True)

        self._set_metric(self.metric)


    @staticmethod
    def olm_diffeo(x):  # 4dim
        x = functions.sym_logm.apply(x)
        return x.tril(-1) + x.tril(-1).transpose(-1, -2) 


    def olm_diffeo_inv(self, x, atol=1e-3, max_iter=1):
            return functions.sym_expm.apply(hol_plus_finder(x, atol=atol,max_iter=max_iter))
    

    def lsm_diffeo(self, x, atol=1e-3, max_iter=1):
        with torch.no_grad():
            diag = spd_scaling_finder(x, atol=atol, max_iter=max_iter).unsqueeze(-1)
        x = diag * x * diag.transpose(-1, -2)
        sym = functions.sym_logm.apply(x)
        return sym
    

    def lsm_diffeo_inv(self,x):
        cov = functions.sym_expm.apply(x)
        std = cov.diagonal(offset=0, dim1=-1, dim2=-2).sqrt().unsqueeze(-1)
        return x / (std @ std.transpose(-1,-2))

    
    def _set_metric(self, metric):
        """Set Correlation manifold metric and transformation"""
        if metric == "olm":
            self.diffeo = self.olm_diffeo
            self.diffeo_inv = self.olm_diffeo_inv

        elif metric == "lsm":
            self.diffeo = self.lsm_diffeo
            self.diffeo_inv = self.lsm_diffeo_inv
        elif metric == "mix":
            self.diffeo = self.lsm_diffeo
            self.diffeo_inv = self.lsm_diffeo_inv
            self.tangent_map = self.olm_diffeo

        else:
            raise ValueError(f"Unsupported metric: {metric}")


    @staticmethod
    def _trans_base(x: torch.Tensor, y: torch.Tensor) -> torch:
        mat_orth = cayley_map(matrix2skew(x))
        y = mat_orth @ y @ mat_orth.transpose(-1, -2)
        return y.tril(-1) + y.tril(-1).transpose(-1, -2)

    def transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._trans_base(self.q_weight, x)
        k = self._trans_base(self.k_weight, x)
        v = self._trans_base(self.v_weight, x)
        return q, k, v

    @staticmethod
    def distance(mat_a, mat_b):
        return torch.norm(mat_a - mat_b, dim=[-1, -2])


    def forward(self, x):
        x = self.diffeo(x)
        q, k, v = self.transform(x)
        atten_energy = self.distance(q.unsqueeze(1), k.unsqueeze(2))
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).transpose(-1, -2)
        x = torch.sum(v.unsqueeze(2) * atten_prob.unsqueeze(-1).unsqueeze(-1), dim=1) # WFM
        if self.metric == "mix":
            x = self.tangent_map(self.diffeo_inv(x))
        return x



class TrilEmbed(nn.Module):
    def __init__(self, ndim, tril=False):
        super().__init__()

        self.tril = tril
        if self.tril:
            ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
            ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
            self.ixs = torch.cat((ixs_diag[None, :].tile((2, 1)), ixs_lower), dim=1)
        else:
            self.ixs = torch.tril_indices(ndim, ndim, offset=-1)  # 只取下三角部分
        self.ndim = ndim

    def forward(self, X: Tensor) -> Tensor:
        return self.embed(X)

    def embed(self, X: Tensor) -> Tensor:

        return X[..., self.ixs[0], self.ixs[1]]



class Signal2Spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self, device='cuda', tr_norm=True):
        super().__init__()
        self.device = device
        self.tr_norm = tr_norm

    def forward(self, x):
        x = x.squeeze()
        x = x - x.mean(dim=-1,keepdim=True)
        cov = x @ x.transpose(-1, -2) / (x.shape[-1] - 1)
        if self.tr_norm:
            cov /= cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True).unsqueeze(-1)
        cov = cov + (1e-5 * torch.eye(cov.shape[-1], device=x.device))
        return cov


class Signal2Cor(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self, power=1.0, device='cuda', tr_norm=True):
        super().__init__()
        self.device = device
        self.tr_norm = tr_norm
        self.signal2spd=Signal2Spd(device=device,tr_norm=tr_norm)

    def forward(self, x):
        cov = self.signal2spd(x)
        std = cov.diagonal(offset=0, dim1=-1, dim2=-2).sqrt().unsqueeze(-1)
        return cov / (std @ std.transpose(-1,-2))


def patch_len(n, epochs):
    base = n // epochs
    remainder = n % epochs
    list_len = [base + 1 if i < remainder else base for i in range(epochs)]
    return list_len


class E2R(nn.Module):
    def __init__(self, epochs, manifold_type='corr', device='cuda', dim=-1):
        super().__init__()
        self.epochs = epochs
        self.device = device
        if manifold_type == 'corr':
            self.signal2manifold = Signal2Cor(device=device)
        elif manifold_type == 'spd':
            self.signal2manifold = Signal2Spd(device=device)
        self.dim = dim
    def forward(self, x):
        # x with shape[bs, ch, time]
        list_patch = patch_len(x.shape[self.dim], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=self.dim))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2manifold(item)

        x = torch.stack(x_list)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2, 3)
        return x



class CorrAttBci2a(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.device = args.device

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 22, (22, 1)).to(self.device),
            nn.BatchNorm2d(22).to(self.device),
            nn.Conv2d(22, args.in_size, (1, 12), padding=(0, 6)).to(self.device),
            nn.BatchNorm2d(args.in_size).to(self.device),
        )

        self.ract1 = E2R(epochs=args.epochs,device=args.device).to(self.device)
        self.att2 = CorrAttention(args.in_size, args.out_size, args.metric, self.device)
        self.tangent = TrilEmbed(args.out_size, tril=True)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(int(args.out_size * (args.out_size + 1) // 2 * args.epochs), 4, bias=True).to(self.device)

    def forward(self, x):
        x = self.cnn(x.to(self.device))
        x = self.ract1(x)
        x = self.att2(x)

        x = self.tangent(x)
        x = self.flat(x)
        x = self.linear(x)
        return x


class CorrAttMamem(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.device = args.device
        dim1 = 125
        # FE
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim1, (8, 1)).to(self.device),
            nn.BatchNorm2d(dim1).to(self.device),
            nn.Conv2d(dim1, args.in_size, (1, 36), padding=(0, 18)).to(self.device),
            nn.BatchNorm2d(args.in_size).to(self.device),
        )

        self.flat = nn.Flatten()
        self.ract1 = E2R(epochs=args.epochs, manifold_type='corr',device=args.device)
        self.att = CorrAttention(args.in_size, args.out_size, metric=args.metric, device=self.device)
        self.tangent = TrilEmbed(args.out_size, tril=False)
        self.linear = nn.Linear(args.out_size * (args.out_size - 1) // 2 * args.epochs, 5, bias=True, device=self.device)


    def forward(self, x):
        x = self.cnn(x.to(self.device))
        x = self.ract1(x)
        x = self.att(x)
        x = self.tangent(x)
        x = self.flat(x)
        x = self.linear(x)
        return x


class CorrAttCha(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.device = args.device
        self.epochs = args.epochs

        dim1 = 23
        # bs, 1, channel, sample
        self.cnn = nn.Sequential(
            nn.Conv2d(args.epochs, dim1*args.epochs, (56, 1), groups=args.epochs).to(self.device),
            nn.BatchNorm2d(dim1*args.epochs).to(self.device),
            nn.Conv2d(dim1*args.epochs, args.in_size*args.epochs, (1, 64), padding=(0, 32), groups=args.epochs).to(self.device),
            nn.BatchNorm2d(args.in_size*args.epochs).to(self.device),
        )

        self.ract1 = E2R(epochs=args.epochs, power=args.power, device=args.device, dim=1).to(self.device)
        self.att2 = CorrAttention(args.in_size, args.out_size, args.epochs, args.metric, self.device)
        self.tangent = TrilEmbed(args.out_size, tril=True)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(args.out_size * (args.out_size + 1) // 2 * args.epochs, 1, bias=True).to(self.device)


    def forward(self, x):
        x = self.cnn(x.repeat(1,self.epochs,1,1).to(self.device))

        x = self.ract1(x)
        x = self.att2(x)
        x = self.tangent(x)
        x = self.flat(x)
        x = self.linear(x)
        return x