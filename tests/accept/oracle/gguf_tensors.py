"""Tensor list from the pinned llama.cpp gguf reader.

Prints a JSON array of {name, shape, ggml_type} in file order.
`shape` is GGUF dimension order (ne[0] first), not the reversed NumPy view.
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
