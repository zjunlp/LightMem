import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist

from src import dist_utils


class InExampleContrastiveLoss:
    """
    Categorization loss: cross_entropy of 1 out of K classes (target labels)
    x.shape=[bsz, hdim], y.shape=[bsz, num_label, hdim]
    """
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, ndim: int = None, *args, **kwargs):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature
        self.ndim = ndim

    def __call__(self, x: Tensor, y: Tensor, reduction: str = 'mean'):
        # print("gather InExampleContrastiveLoss")
        if torch.distributed.is_initialized():
            x = dist_utils.dist_gather(x)
            y = dist_utils.dist_gather(y)
        bsz, ndim = x.size(0), x.size(1)
        target = torch.zeros(bsz, dtype=torch.long, device=x.device)
        if self.ndim:
            ndim = self.ndim
            x = x[:, :ndim]
            y = y[:, :ndim]
        logits = torch.einsum('bod,bsd->bs', x.view(bsz, 1, ndim), y.view(bsz, -1, ndim)) * self.temperature
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, target, reduction=reduction)
        loss_detail = {"logits": logits, "labels": target, "preds": preds}
        return loss, loss_detail


class SimpleContrastiveLoss:
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, *args, **kwargs):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        # print("gather SimpleContrastiveLoss")
        if target is None:
            assert x.size(0) * self.target_per_qry == y.size(0)
            target = torch.arange(0, y.size(0), step=self.target_per_qry, dtype=torch.long, device=x.device)
        logits = torch.matmul(x, y.transpose(0, 1)) * self.temperature
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, target, reduction=reduction)
        loss_detail = {"logits": logits, "labels": target, "preds": preds}
        return loss, loss_detail


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, *args, **kwargs):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."

        super().__init__(n_hard_negatives=n_hard_negatives, temperature=temperature)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        # print("gather DistributedContrastiveLoss")
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)

        return super().__call__(dist_x, dist_y, **kwargs)

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


LossName2LossCls = {
    "inexample_contrastive": InExampleContrastiveLoss,
    "inbatch_contrastive": SimpleContrastiveLoss,
    "distributed_inbatch_contrastive": DistributedContrastiveLoss,
}