
import torch.nn as nn
import torch
import torch.nn.functional as F

def ib_loss(input_values, ib):
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., num_classes=3):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.num_classes = num_classes
        self.weight = weight
        self.epsilon = 0.001

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)), 1)
        ib = grads * features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon) #变乘除法， 倒过来
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)

def ib_focal_loss(input_values, ib, gamma):
    p = torch.exp(-input_values)
    loss = (1-p) ** gamma * input_values * ib
    return loss.mean()

class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., gamma=0., num_classes=3):
        super(IB_FocalLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.apsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)), 1)
        features = features.reshape((-1))
        ib = grads*(features)
        ib = self.alpha / (ib + self.apsilon)
        return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)