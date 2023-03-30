from typing import Union, Tuple, Optional

from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import glorot, zeros


class GatConv(MessagePassing):
    """
    Args:
        add later.
    """
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        dim_Y: int = 25,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = 128,
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super(GatConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_e_v = Linear(edge_dim + dim_Y, heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.lin_edge = Linear(in_channels + out_channels + edge_dim + dim_Y, edge_dim, bias=bias)
        self.lin_edge_2 = Linear(edge_dim, edge_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_e_v.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_edge_2.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, pbc_index, ylm):
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        # propagate
        edge_ylm = torch.cat([edge_attr, ylm], dim=-1)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_ylm)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        out = out[pbc_index]

        # update edge
        xi, xj = out[edge_index]
        edge_out = torch.cat([xi, xj, edge_attr, ylm], dim=-1)
        edge_out = F.hardswish(self.lin_edge(edge_out))
        edge_out = self.lin_edge_2(edge_out)

        self._alpha = None

        return out, edge_out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = self.lin_e_v(edge_attr)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        x += edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, '
                f'edge_dim={self.edge_dim}, heads={self.heads}, dropout={self.dropout})')


class Lcmp(torch.nn.Module):
    """
    Args:
        channels (Union[int, Tuple[int, int]]): the dimension of node feature.
        dim_e: (int): the dimension of egde feature.
        dim_Y: (int): the number of Ylm.
    """
    def __init__(self, channels=64, dim_e=128, dim_Y=64, dim_ham=1, heads=1, dropout=0.0, bias=True):
        super(Lcmp, self).__init__()

        self.embedding = torch.nn.Embedding(100, channels)
        self.lin_ylm = YlmConv(out_dim=dim_Y//2)  # 分别计算 ylm_i 和 ylm_j，再拼起来

        self.interactions = ModuleList()
        for _ in range(5):
            block = GatConv(in_channels=channels, out_channels=channels, dim_Y=dim_Y, heads=heads, dropout=dropout)
            self.interactions.append(block)

        # update edge attribute
        self.lin_edge_1 = Linear(channels*2+dim_e+dim_Y, 256, bias=bias)
        self.lin_edge_2 = Linear(256, dim_ham, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_edge_1.reset_parameters()
        self.lin_edge_2.reset_parameters()

    def forward(self, data, rev=False):
        h = self.embedding(data.x)
        # print(data.ylm)
        # print(data.ylm[0])
        ylm_j = self.lin_ylm(data.ylm[0])
        ylm_i = self.lin_ylm(data.ylm[1])
        ylm = torch.cat([ylm_j, ylm_i], dim=1)
        edge_index, edge_attr, pbc_index = data.edge_index, data.edge_attr, data.pbc_index

        for interaction in self.interactions:
            h_out, edge_attr_out = interaction(h, edge_index, edge_attr, pbc_index, ylm)
            h = h + h_out
            edge_attr = edge_attr + edge_attr_out

        xi, xj = h[edge_index]

        edge_out = torch.cat([xi, xj, edge_attr, ylm], dim=-1)
        edge_out = F.hardswish(self.lin_edge_1(edge_out))
        edge_out = self.lin_edge_2(edge_out)

        if rev:
            edge_out = torch.where(data.rev.view(-1, 1), edge_out, torch.zeros_like(edge_out))
        else:
            edge_out = torch.where(~data.rev.view(-1, 1), edge_out, torch.zeros_like(edge_out))

        return edge_out


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class YlmConv(torch.nn.Module):
    """
    用TextCNN处理局域坐标下的球谐函数。参考：https://www.jb51.net/article/212414.htm
    """
    def __init__(self, in_dim=25, out_dim=64, num_kernel=64):
        super().__init__()
        # self.conv = torch.nn.Conv2d(1, 64, (5, in_dim))
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1, num_kernel, (K, in_dim)) for K in (2, 3, 4)])
        self.lin = Linear(num_kernel*3, out_dim, bias=True)
        # self.convs.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x):
        # data.ylm.shape = (num_edge, num_nbr, 25)
        x = x.unsqueeze(1)
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x, 1)
        out = self.lin(x)
        return out


if __name__ == '__main__':
    data = torch.load('../dataset/graphene/0.pt')
    print(data)
    # for key in data.keys:
    #     v = data.__getitem__(key)
    #     res = torch.isnan(v)
    #     if res.any():
    #         print(key)
    #         print(v[res])

    model = Lcmp()
    out = model(data)
    # print(out)


