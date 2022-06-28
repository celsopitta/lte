import tinycudann as tcnn

from models import register


@register('mrh')
def make_rcan(n_input_dims, args):
   
    return tcnn.Encoding(n_input_dims, args)