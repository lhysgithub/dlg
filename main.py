# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)


gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tt(gt_data[0].cpu()))
plt.savefig("original_data.png")

from models.vision import LeNet, weights_init, ResNet18, weights_init_for_resnet
# net = LeNet().to(device)
net = ResNet18().to(device)

torch.manual_seed(1234)

# net.apply(weights_init)
net.apply(weights_init_for_resnet)
criterion = cross_entropy_for_onehot
net.eval()
# compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))
plt.savefig("dummy_data.png")

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        # grad_diff = 0
        # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
        #     # grad_diff += (torch.abs(gx - gy)).sum()
        #     grad_diff += ((gx - gy)**2).sum()
        # grad_diff = sum((10000*(dummy_g - origin_g)**2).sum() for dummy_g, origin_g in zip(dummy_dy_dx, original_dy_dx))
        grad_diff = sum(((dummy_g - origin_g) ** 2).sum() for dummy_g, origin_g in zip(dummy_dy_dx, original_dy_dx))
        # grad_diff = sum((torch.abs(dummy_g - origin_g)).sum() for dummy_g, origin_g in zip(dummy_dy_dx, original_dy_dx))
        # grad_diff = torch.log(grad_diff)
        grad_diff.backward()
        # print(iters, "%f" % grad_diff)
        # print(
        #     f"dummy_data.grad max: {dummy_data.grad.max()} min: {dummy_data.grad.min()} mean: {dummy_data.grad.mean()}")
        # print(
        #     f"dummy_label.grad max: {dummy_label.grad.max()} min: {dummy_label.grad.min()} mean: {dummy_label.grad.mean()}")


        
        return grad_diff
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%f" % current_loss.item())
        print(
            f"dummy_data.grad max: {dummy_data.grad.max()} min: {dummy_data.grad.min()} mean: {dummy_data.grad.mean()}")
        print(
            f"dummy_label.grad max: {dummy_label.grad.max()} min: {dummy_label.grad.min()} mean: {dummy_label.grad.mean()}")
        history.append(tt(dummy_data[0].cpu()))
    optimizer.step(closure)


plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i*10))
    plt.axis('off')
plt.savefig("process.png")
# plt.show()
