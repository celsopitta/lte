import torch.nn as nn
import commentjson as json
import tinycudann as tcnn

from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

@register('tiny-mlp')
def make_tiny_mlp(args):
    
    ''''
    config = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation":"None",
    "n_neurons": args.n_neurons,
    "n_hidden_layers": args.n_hidden_layers
    }
    '''

    return tcnn.Network(args.in_dim, args.out_dim, args)

