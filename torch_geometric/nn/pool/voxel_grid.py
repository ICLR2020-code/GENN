import torch
from torch_cluster import grid_cluster
from torch_geometric.utils.repeat import repeat


def voxel_grid(pos, batch, size, start=None, end=None):
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (float or [float]): Size of a voxel (in each dimension).
        start (float or [float], optional): Start coordinates of the grid (in
            each dimension). If set to :obj:`None`, will be set to the minimum
            coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float], optional): End coordinates of the grid (in
            each dimension). If set to :obj:`None`, will be set to the maximum
            coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    num_nodes, dim = pos.size()

    size, start, end = repeat(size, dim), repeat(start, dim), repeat(end, dim)

    pos = torch.cat([pos, batch.unsqueeze(-1).type_as(pos)], dim=-1)
    size = size + [1]
    start = None if start is None else start + [0]
    end = None if end is None else end + [batch.max().item()]

    size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
    if start is not None:
        start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
    if end is not None:
        end = torch.tensor(end, dtype=pos.dtype, device=pos.device)

    return grid_cluster(pos, size, start, end)
