# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import numpy as np
import torch
import torch.nn as nn



# (number of input_node, output_node)
num_nodes = [(256, 128), (128, 64), (64, 16), (16, 0)]
DECODER = {
    'same': [(64, 128), (128, 256), (256, 1193), (1193, 0)]
}

class MLP(nn.Module):
    def __init__(self, features, num_classes, decoding=False):
        super(MLP, self).__init__()
        self.features = features
        latent_dim = 16
        self.top_layer = nn.Linear(latent_dim, num_classes)
        self._initialize_weights()
        self.decoding = decoding
        # train or val
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 68),
            nn.Tanh())
        # test
        # self.decoder = make_layers_features(DECODER['same'], 16, bn=True)
        # self.decoder = self.decoder[:-2]

    def forward(self, x):
        # test
        # # x = self.features(x)
        # if self.top_layer and self.decoder:
        #     y2 = self.decoder(x)
        #     return y2
        # elif self.top_layer and not self.decoder:
        #     y1 = self.top_layer(x)
        #     return y1
        # else:
        #     return x

        # original
        x = self.features(x)
        if self.top_layer and self.decoding:
            y2 = self.decoder(x)
            return y2
        elif self.top_layer and not self.decoding:
            y1 = self.top_layer(x)
            return y1
        else:
            return x
        if self.top_layer:
            y1 = self.top_layer(x)
            return y1
        else:
            return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers_features(num_node, input_dim, bn):
    layers = []
    for v in num_node:
        layer = nn.Linear(input_dim, v[0])
        if bn:
            layers += [layer.cuda(), nn.BatchNorm1d(v[0]), nn.ReLU(inplace=True)]
        else:
            layers += [layer.cuda(), nn.ReLU(inplace=True)]

        input_dim = v[0]
    return nn.Sequential(*layers)


def mlp(input_dim=1193, bn=True, output_dim=4):
    model = MLP(make_layers_features(num_nodes, input_dim, bn=bn), output_dim)
    return model
