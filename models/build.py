#!/usr/bin/env python3
# encoding: utf-8
from .unet import Unet_module, UNet3D
from .AdaMMKD import AdaMMKD
from .HiKINet import HiKINet

def build_model(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = UNet3D(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = UNet3D(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing

def build_AdaMMKD(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = AdaMMKD(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = AdaMMKD(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing

def build_HiKINet(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model    = HiKINet(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, kd_loss_type='mse', fusion_mode='moe')
    return model