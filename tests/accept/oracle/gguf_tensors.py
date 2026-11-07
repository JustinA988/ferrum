"""Tensor list from the pinned llama.cpp GGUF reader.

Prints a JSON array of {name, shape, ggml_type} in file order. `shape` is
GGUF dimension order, with ne[0] first, not the reversed NumPy view. The
GGUF acceptance test compares this list with `ferrum_gguf::load`. The script
is an offline check. The server does not import it.
"""

import json
import sys

from gguf import GGUFReader

reader = GGUFReader(sys.argv[1])
rows = []
for tensor in reader.tensors:
    rows.append(
        {
            "name": tensor.name,
            "shape": [int(dim) for dim in tensor.shape.tolist()],
            "ggml_type": int(tensor.tensor_type),
        }
    )
json.dump(rows, sys.stdout)
sys.stdout.write("\n")
