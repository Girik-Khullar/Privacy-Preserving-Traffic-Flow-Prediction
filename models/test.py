#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net, datatest, args):
    net.eval()
    # Testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    num_samples = len(datatest)
    
    with torch.no_grad():
        for data, target in data_loader:
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            outputs = net(data)
            # Calculate loss
            test_loss += F.mse_loss(outputs, target).item()  # Using MSE loss for regression
            
            # Calculate accuracy based on a threshold or tolerance level
            # For example, if the absolute difference between prediction and target is less than a threshold, count it as correct
            correct += torch.sum(torch.abs(outputs - target) < args.threshold).item()

    test_loss /= num_samples
    accuracy = 100.0 * correct / num_samples
    
    if args.verbose:
        print('\nTest set: Average loss: {:.4f}\nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, num_samples, accuracy))    
    return accuracy, test_loss

