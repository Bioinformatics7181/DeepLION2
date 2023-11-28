# -------------------------------------------------------------------------
# Name: loss.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: The loss functions for models.
# -------------------------------------------------------------------------

import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, w=None, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        if self.w is None:
            # SCE
            # CCE
            ce = self.cross_entropy(pred, labels)
            # RCE
            pred = F.softmax(pred, dim=1)
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
            label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
            rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
            # Loss
            loss = self.alpha * ce + self.beta * rce.mean()
        else:
            # WSCE
            pred = F.softmax(pred, dim=1)
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
            label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
            loss = self.alpha * (-1 * torch.sum(self.w * label_one_hot * torch.log(pred), dim=1)) + \
                   self.beta * (-1 * torch.sum(self.w * pred * torch.log(label_one_hot), dim=1))
            loss = loss.mean()
        return loss


class GCELoss(torch.nn.Module):
    def __init__(self, q=0.7, w=None, num_classes=10):
        super(GCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q = q
        self.w = w
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        if self.w is None:
            # GCE
            loss = (1 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        else:
            # WGCE
            loss = (1 - torch.pow(torch.sum(self.w * label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class CELoss(torch.nn.Module):
    def __init__(self, w=None, num_classes=10):
        super(CELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w = w
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        if self.w is None:
            # CE
            loss = (-1 * torch.sum(label_one_hot * torch.log(pred), dim=1))
        else:
            # WCE
            loss = (-1 * torch.sum(self.w * label_one_hot * torch.log(pred), dim=1))
        return loss.mean()


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.05, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, labels):
        alpha = int(torch.sum(labels)) / labels.shape[0] + self.alpha
        w = torch.Tensor([alpha, 1 - alpha])
        ce = torch.nn.CrossEntropyLoss(weight=w)
        logp = ce(pred, labels)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss


class DSCLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == "mean":
            return loss.mean()
