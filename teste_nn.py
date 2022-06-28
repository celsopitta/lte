import commentjson as json
import tinycudann as tcnn
import torch

with open("data/config_hash.json") as f:
	config = json.load(f)

# Option 1: efficient Encoding+Network combo.
model = tcnn.NetworkWithInputEncoding(
	n_input_dims, n_output_dims,
	config["encoding"], config["network"]
)

# Option 2: separate modules. Slower but more flexible.
encoding = tcnn.Encoding(n_input_dims, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, n_output_dims, config["network"])
model = torch.nn.Sequential(encoding, network)