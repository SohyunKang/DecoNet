# decoding process should be added
import torch.nn as nn


# number of input_node
num_nodes = [256, 128, 64, 16]

class MLP(nn.Module):
    def __init__(self, features, num_classes, decoding=False):
        super(MLP, self).__init__()
        self.features = features
        latent_dim = 16
        self.top_layer = nn.Linear(latent_dim, num_classes)
        self._initialize_weights()
        self.decoding = decoding

    def forward(self, x):
        # original
        x = self.features(x)
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
        layer = nn.Linear(input_dim, v)
        if bn:
            layers += [layer.cuda(), nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
        else:
            layers += [layer.cuda(), nn.ReLU(inplace=True)]

        input_dim = v
    return nn.Sequential(*layers)


def mlp(input_dim=1193, bn=True, output_dim=4):
    model = MLP(make_layers_features(num_nodes, input_dim, bn=bn), output_dim)
    return model
