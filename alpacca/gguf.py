# Alpacca - GGUF file reader/writer, implemented from the format spec.
# (GGUF is the model file format defined by the ggml project; this is an
# independent pure-Python implementation, no ggml code involved.)
# MIT License. See LICENSE.
"""Read and write GGUF model files.

Reader: parses the header, metadata key/values and tensor table, and
memory-maps tensor data. Writer: enough of the format to produce small,
valid models for tests (F32/F16/Q8_0/Q4_0 tensors).
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}

# ggml tensor data types (subset we know how to handle lives in quants.py)
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}
GGML_TYPE_IDS = {v: k for k, v in GGML_TYPE_NAMES.items()}

# (block_elements, block_bytes) for the types we can read; see quants.py
GGML_BLOCK_INFO = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210),
    "I8": (1, 1), "I16": (1, 2), "I32": (1, 4), "F64": (1, 8),
}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # ggml order: shape[0] is the contiguous dim
    dtype: str              # "F32", "Q4_K", ...
    offset: int             # relative to the data section

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        if self.dtype not in GGML_BLOCK_INFO:
            raise ValueError(f"unsupported tensor type {self.dtype} for {self.name}")
        block_n, block_b = GGML_BLOCK_INFO[self.dtype]
        if self.n_elements % block_n != 0:
            raise ValueError(f"tensor {self.name} not a multiple of block size")
        return self.n_elements // block_n * block_b


@dataclass
class GGUFFile:
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    data_start: int = 0
    _mm: mmap.mmap | None = None
    _fh: BinaryIO | None = None

    # -- reading ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> "GGUFFile":
        f = cls(Path(path))
        f._fh = open(f.path, "rb")
        f._mm = mmap.mmap(f._fh.fileno(), 0, access=mmap.ACCESS_READ)
        f._parse()
        return f

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _parse(self) -> None:
        mm = self._mm
        assert mm is not None
        if mm[:4] != GGUF_MAGIC:
            raise ValueError(f"{self.path} is not a GGUF file")
        pos = 4
        version, = struct.unpack_from("<I", mm, pos)
        pos += 4
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = struct.unpack_from("<QQ", mm, pos)
        pos += 16

        def read_string(p: int) -> tuple[str, int]:
            ln, = struct.unpack_from("<Q", mm, p)
            p += 8
            s = mm[p:p + ln].decode("utf-8", errors="replace")
            return s, p + ln

        def read_value(vtype: int, p: int) -> tuple[Any, int]:
            if vtype in _SCALAR_FMT:
                fmt = _SCALAR_FMT[vtype]
                v, = struct.unpack_from(fmt, mm, p)
                return v, p + struct.calcsize(fmt)
            if vtype == T_BOOL:
                return mm[p] != 0, p + 1
            if vtype == T_STRING:
                return read_string(p)
            if vtype == T_ARRAY:
                etype, = struct.unpack_from("<I", mm, p)
                count, = struct.unpack_from("<Q", mm, p + 4)
                p += 12
                # fast path for big scalar arrays (token scores etc.)
                if etype in _SCALAR_FMT:
                    fmt = _SCALAR_FMT[etype]
                    size = struct.calcsize(fmt)
                    vals = list(struct.unpack_from(f"<{count}{fmt[1]}", mm, p))
                    return vals, p + count * size
                out = []
                for _ in range(count):
                    v, p = read_value(etype, p)
                    out.append(v)
                return out, p
            raise ValueError(f"unknown GGUF metadata type {vtype}")

        for _ in range(n_kv):
            key, pos = read_string(pos)
            vtype, = struct.unpack_from("<I", mm, pos)
            pos += 4
            value, pos = read_value(vtype, pos)
            self.metadata[key] = value

        for _ in range(n_tensors):
            name, pos = read_string(pos)
            n_dims, = struct.unpack_from("<I", mm, pos)
            pos += 4
            dims = struct.unpack_from(f"<{n_dims}Q", mm, pos)
            pos += 8 * n_dims
            dtype_id, offset = struct.unpack_from("<IQ", mm, pos)
            pos += 12
            dtype = GGML_TYPE_NAMES.get(dtype_id, f"UNKNOWN_{dtype_id}")
            self.tensors[name] = TensorInfo(name, tuple(dims), dtype, offset)

        align = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        self.data_start = (pos + align - 1) // align * align

    def tensor_bytes(self, name: str) -> memoryview:
        """Raw (possibly quantized) bytes of a tensor, zero-copy."""
        info = self.tensors[name]
        start = self.data_start + info.offset
        assert self._mm is not None
        return memoryview(self._mm)[start:start + info.n_bytes]

    # convenience accessors --------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))


# ---- writing (test fixtures and tooling) --------------------------------

class GGUFWriter:
    """Minimal GGUF v3 writer. Tensors are given as flat lists of floats in
    ggml layout and stored as F32/F16/Q8_0/Q4_0."""

    def __init__(self, path: str | Path, architecture: str):
        self.path = Path(path)
        self.kv: list[tuple[str, int, Any]] = []
        self.tensors: list[tuple[str, tuple[int, ...], str, bytes]] = []
        self.add("general.architecture", T_STRING, architecture)

    def add(self, key: str, vtype: int, value: Any) -> None:
        self.kv.append((key, vtype, value))

    def add_array(self, key: str, etype: int, values: list) -> None:
        self.kv.append((key, T_ARRAY, (etype, values)))

    def add_tensor(self, name: str, shape: tuple[int, ...], values: list[float],
                   dtype: str = "F32") -> None:
        from . import quants
        n = 1
        for d in shape:
            n *= d
        if len(values) != n:
            raise ValueError(f"{name}: {len(values)} values for shape {shape}")
        if dtype == "F32":
            data = struct.pack(f"<{n}f", *values)
        elif dtype == "F16":
            data = struct.pack(f"<{n}e", *values)
        elif dtype == "Q8_0":
            data = quants.quantize_q8_0(values)
        elif dtype == "Q4_0":
            data = quants.quantize_q4_0(values)
        else:
            raise ValueError(f"writer does not support {dtype}")
        self.tensors.append((name, shape, dtype, data))

    def add_raw_tensor(self, name: str, shape: tuple[int, ...], dtype: str,
                       data: bytes) -> None:
        """Add pre-encoded GGUF tensor bytes.

        This is intended for tests that need a valid container for formats the
        writer cannot quantize from float values yet.
        """
        if dtype not in GGML_BLOCK_INFO:
            raise ValueError(f"writer does not support raw tensor type {dtype}")
        n = 1
        for d in shape:
            n *= d
        block_n, block_b = GGML_BLOCK_INFO[dtype]
        if n % block_n:
            raise ValueError(f"{name}: element count is not a multiple of {block_n}")
        expected = n // block_n * block_b
        if len(data) != expected:
            raise ValueError(f"{name}: {len(data)} bytes for {dtype}, expected {expected}")
        self.tensors.append((name, shape, dtype, bytes(data)))

    @staticmethod
    def _pack_string(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _pack_value(self, vtype: int, value: Any) -> bytes:
        if vtype in _SCALAR_FMT:
            return struct.pack(_SCALAR_FMT[vtype], value)
        if vtype == T_BOOL:
            return struct.pack("<B", 1 if value else 0)
        if vtype == T_STRING:
            return self._pack_string(str(value))
        if vtype == T_ARRAY:
            etype, values = value
            out = struct.pack("<IQ", etype, len(values))
            for v in values:
                out += self._pack_value(etype, v)
            return out
        raise ValueError(f"cannot pack type {vtype}")

    def write(self) -> None:
        align = DEFAULT_ALIGNMENT
        header = bytearray()
        header += GGUF_MAGIC
        header += struct.pack("<IQQ", GGUF_VERSION, len(self.tensors), len(self.kv))
        for key, vtype, value in self.kv:
            header += self._pack_string(key)
            header += struct.pack("<I", vtype)
            header += self._pack_value(vtype, value)

        offset = 0
        blobs: list[bytes] = []
        for name, shape, dtype, data in self.tensors:
            header += self._pack_string(name)
            header += struct.pack("<I", len(shape))
            header += struct.pack(f"<{len(shape)}Q", *shape)
            header += struct.pack("<IQ", GGML_TYPE_IDS[dtype], offset)
            padded = (len(data) + align - 1) // align * align
            blobs.append(data + b"\x00" * (padded - len(data)))
            offset += padded

        data_start = (len(header) + align - 1) // align * align
        with open(self.path, "wb") as f:
            f.write(header)
            f.write(b"\x00" * (data_start - len(header)))
            for blob in blobs:
                f.write(blob)
