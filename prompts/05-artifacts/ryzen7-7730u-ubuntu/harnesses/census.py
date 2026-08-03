"""Census a GGUF's tensor dtypes and the distinct decode matvec shapes.

Reads the header only - no weights are loaded - so this is milliseconds even
on an 8B file.
"""
import sys
from collections import Counter, defaultdict

from alpaccaroo.gguf import GGUFFile
from alpaccaroo import store

THRESHOLD = 131072  # ALPACCAROO_SERIAL_MATVEC_ELEMS default


def census(ref: str) -> None:
    parsed = store.parse_model_ref(store.resolve_model_input(ref))
    local = store.find_local(parsed)
    if local is None:
        raise SystemExit(f"{ref} is not installed")
    path = local.model_path
    print(f"===== {ref}  ({path})")
    with GGUFFile.open(path) as g:
        arch = g.architecture
        n_layer = g.get(f"{arch}.block_count")
        embd = g.get(f"{arch}.embedding_length")
        n_head = g.get(f"{arch}.attention.head_count")
        n_kv = g.get(f"{arch}.attention.head_count_kv")
        ff = g.get(f"{arch}.feed_forward_length")
        ctx = g.get(f"{arch}.context_length")
        print(f"arch={arch} layers={n_layer} embd={embd} heads={n_head}/{n_kv} "
              f"ff={ff} ctx_train={ctx}")

        by_dtype = Counter()
        bytes_by_dtype = Counter()
        for t in g.tensors.values():
            by_dtype[t.dtype] += 1
            bytes_by_dtype[t.dtype] += t.n_bytes
        total = sum(bytes_by_dtype.values())
        print("  dtype census:")
        for d, n in by_dtype.most_common():
            print(f"    {d:<10} {n:>4} tensors  {bytes_by_dtype[d]/2**20:>9.1f} MiB"
                  f"  {100*bytes_by_dtype[d]/total:>5.1f}%")

        # distinct 2-D weight shapes = the decode matvecs
        shapes = defaultdict(lambda: [0, set(), set()])
        for name, t in g.tensors.items():
            if len(t.shape) != 2:
                continue
            key = (t.shape[1], t.shape[0])  # ggml shape[0] is the contiguous (column) dim
            shapes[key][0] += 1
            shapes[key][1].add(t.dtype)
            role = name.split(".")[-2] if "." in name else name
            shapes[key][2].add(role)
        print("  distinct 2-D shapes (elements = rows*cols):")
        for key in sorted(shapes, key=lambda k: k[0] * k[1]):
            n, dts, roles = shapes[key]
            elems = key[0] * key[1]
            mark = "  <-- CROSSES narrow threshold" if elems <= THRESHOLD else ""
            print(f"    {key[0]:>6} x {key[1]:<6} = {elems:>10}  n={n:<3} "
                  f"{'+'.join(sorted(dts)):<12} {','.join(sorted(roles))}{mark}")


if __name__ == "__main__":
    for ref in sys.argv[1:]:
        census(ref)
        print()
