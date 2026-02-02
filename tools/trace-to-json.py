#!/usr/bin/env python3
"""Convert keren-sim --trace output to JSON for the visualizer.

Usage: python3 trace-to-json.py /tmp/keren-trace

Reads index.csv and .npy files, outputs JSON mapping probe IDs to values.
"""

import csv
import json
import sys
import os

def read_npy_simple(path):
    """Read a NumPy .npy file and return the data as a nested list."""
    import struct

    with open(path, 'rb') as f:
        magic = f.read(6)
        if magic[:6] != b'\x93NUMPY':
            raise ValueError(f"Not a NumPy file: {path}")

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]

        if major == 1:
            header_len = struct.unpack('<H', f.read(2))[0]
        else:
            header_len = struct.unpack('<I', f.read(4))[0]

        header = f.read(header_len).decode('latin1').strip()
        # Parse header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 2)}
        header_dict = eval(header)  # safe enough for npy headers
        descr = header_dict['descr']
        shape = header_dict['shape']
        fortran = header_dict.get('fortran_order', False)

        # Map dtype to struct format
        dtype_map = {
            '<f4': ('f', 4), '>f4': ('f', 4),
            '<f8': ('d', 8), '>f8': ('d', 8),
            '<i4': ('i', 4), '>i4': ('i', 4),
            '<i8': ('q', 8), '>i8': ('q', 8),
            '<i2': ('h', 2), '>i2': ('h', 2),
            '<u4': ('I', 4), '>u4': ('I', 4),
            '|b1': ('?', 1), '|u1': ('B', 1), '|i1': ('b', 1),
        }

        if descr not in dtype_map:
            return f"<unsupported dtype: {descr}>"

        fmt_char, item_size = dtype_map[descr]
        byte_order = '>' if descr.startswith('>') else '<'

        total = 1
        for s in shape:
            total *= s

        data = []
        for _ in range(total):
            raw = f.read(item_size)
            val = struct.unpack(f'{byte_order}{fmt_char}', raw)[0]
            data.append(val)

        # Reshape
        return reshape(data, shape)

def reshape(flat, shape):
    if len(shape) == 0:
        return flat[0] if flat else None
    if len(shape) == 1:
        return flat
    size = 1
    for s in shape[1:]:
        size *= s
    return [reshape(flat[i*size:(i+1)*size], shape[1:]) for i in range(shape[0])]

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace-dir>", file=sys.stderr)
        sys.exit(1)

    trace_dir = sys.argv[1]
    index_path = os.path.join(trace_dir, 'index.csv')

    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    result = {}
    with open(index_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            probe_id, dtype, npy_path = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                value = read_npy_simple(npy_path)
                result[probe_id] = {"type": dtype, "value": value}
            except Exception as e:
                result[probe_id] = {"type": dtype, "error": str(e)}

    json.dump(result, sys.stdout, indent=2)
    print()

if __name__ == '__main__':
    main()
