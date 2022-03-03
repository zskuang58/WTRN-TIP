from . import wavelet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import MeanShift

class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()
        
        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice21 = torch.nn.Sequential()
        self.pool1 = torch.nn.Sequential()
        self.slice22 = torch.nn.Sequential()
        self.slice31 = torch.nn.Sequential()
        self.pool2 = torch.nn.Sequential()
        self.slice32 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice21.add_module(str(x), vgg_pretrained_features[x])
        self.pool1.add_module(str(4),wavelet.WavePool(64))
        for x in range(5, 7):
            self.slice22.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 9):
            self.slice31.add_module(str(x), vgg_pretrained_features[x])
        self.pool2.add_module(str(9),wavelet.WavePool(128))
        for x in range(10, 12):
            self.slice32.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice21.parameters():
                param.requires_grad = requires_grad
            #for param in self.pool1.parameters():
            #    param.requires_grad = requires_grad
            for param in self.slice22.parameters():
                param.requires_grad = requires_grad
            for param in self.slice31.parameters():
                param.requires_grad = requires_grad
            #for param in self.pool2.parameters():
            #    param.requires_grad = requires_grad
            for param in self.slice32.parameters():
                param.requires_grad = requires_grad
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        skips = {}
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice21(x)
        skips['conv1'] = x
        LL,LH,HL,HH = self.pool1(x)
        skips['pool1'] = [LH, HL, HH]
        x = self.slice22(LL)
        x_lv2 = x
        x = self.slice31(x)
        skips['conv2'] = x
        LL,LH,HL,HH = self.pool2(x)
        skips['pool2'] = [LH,HL,HH]
        x = self.slice32(LL)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3, skips



if __name__ == '__main__':
    model = LTE()
    #_, out = LET(x, maps)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)
