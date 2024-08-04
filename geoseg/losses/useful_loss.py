import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
from torchvision.transforms import Resize, InterpolationMode


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        # dice_loss = 1 - ((2. * (boundary_pre * boundary_targets).sum(1) + 1.0) /
        #                  (boundary_pre.sum(1) + boundary_targets.sum(1) + 1.0))
        # dice_loss = dice_loss.mean()
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        loss = (self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor) / (self.edge_factor+1)
        return loss


class OHEM_CELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss
    

# class UMamabaLoss(nn.Module):

#     def __init__(self, ignore_index=255):
#         super().__init__()
#         self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
#                                    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
#         self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
#         # self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         # self.eps = 1e-8

#     def forward(self, logits, labels):
#         if self.training and len(logits) == 2:
#             _, H, W = labels.shape
#             logit_main, logit_aux = logits
#             labels_16x16 = Resize((H // 32, W // 32), interpolation=InterpolationMode.NEAREST, antialias=True)(labels)
#             loss = self.main_loss(logit_main, labels) + 0.8 * self.aux_loss(logit_aux, labels_16x16)
#             # weights = nn.ReLU()(self.weights)
#             # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#             # loss = fuse_weights[0] * self.main_loss(logit_main, labels) + fuse_weights[1] * self.aux_loss(logit_aux, labels_16x16)
#             # print(fuse_weights[0].data.cpu().numpy(), fuse_weights[1].data.data.cpu().numpy())
#         else:
#             loss = self.main_loss(logits, labels)

#         return loss

class UMambaLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        # self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.eps = 1e-8

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            _, H, W = labels.shape
            logit_main, logit_aux = logits
            labels_32x = Resize((H // 32, W // 32), interpolation=InterpolationMode.NEAREST, antialias=True)(labels)
            loss = 0.5 * self.main_loss(logit_main, labels) + 0.5 * self.aux_loss(logit_aux, labels_32x)
            # weights = nn.ReLU()(self.weights)
            # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
            # loss = fuse_weights[0] * self.main_loss(logit_main, labels) + fuse_weights[1] * self.aux_loss(logit_aux, labels_16x16)
            # print(fuse_weights[0].data.cpu().numpy(), fuse_weights[1].data.data.cpu().numpy())
        elif self.training and len(logits) == 3:
            _, H, W = labels.shape
            logit_main, logit_aux16, logit_aux32 = logits
            labels_32x = Resize((H // 32, W // 32), interpolation=InterpolationMode.NEAREST, antialias=True)(labels)
            labels_16x = Resize((H // 16, W // 16), interpolation=InterpolationMode.NEAREST, antialias=True)(labels)
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux32, labels_32x) + 0.4 * self.aux_loss(logit_aux16, labels_16x)
        else:
            loss = self.main_loss(logits, labels)

        return loss


if __name__ == '__main__':
    targets = torch.randint(low=0, high=2, size=(2, 16, 16))
    logits = torch.randn((2, 2, 16, 16))
    # print(targets)
    model = EdgeLoss()
    loss = model.compute_edge_loss(logits, targets)

    print(loss)