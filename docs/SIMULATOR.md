# keren-sim — StableHLO Simulator

`keren-sim` is a command-line tool that interprets StableHLO MLIR programs using the StableHLO reference interpreter. It supports input tensor values, JSON output, and per-op trace instrumentation.

## Building

```bash
cd keren
mkdir -p build && cd build
cmake .. -DLLVM_DIR=/path/to/llvm -DStablehlo_DIR=/path/to/stablehlo
cmake --build . --target keren-sim
```

## Usage

```
keren-sim [options] <input.mlir>
```

### Options

| Flag | Description |
|---|---|
| `--entry=<name>` | Entry function name (default: `main`) |
| `--json` | Output results as JSON |
| `--input='<json>'` | Input tensor as JSON array (repeatable, one per argument) |
| `--input-file=<path>` | JSON file containing array of input tensors |
| `--trace=<dir>` | Enable probe instrumentation, write per-op `.npy` files to directory |
| `--lower-to-linalg` | Lower StableHLO to Linalg before interpreting (prints IR, does not evaluate) |

### Examples

**Run a function with no arguments:**

```bash
keren-sim test.mlir
# Result 0: tensor<2x2xi32> { [[11, 22], [33, 44]] }
```

**Provide input values inline:**

```bash
keren-sim --json \
  --input='[[1,2],[3,4]]' \
  --input='[[10,20],[30,40]]' \
  test/stablehlo/add.mlir
```

**Provide input values from a file:**

```bash
keren-sim --json --input-file=examples/gpt2_inputs.json examples/gpt2.mlir
```

**Trace intermediate values:**

```bash
mkdir -p /tmp/trace
keren-sim --json --trace=/tmp/trace \
  --input='[[1,2],[3,4]]' \
  --input='[[10,20],[30,40]]' \
  test/stablehlo/add.mlir
```

This creates `/tmp/trace/index.csv` and `.npy` files for each op's output. Convert to JSON:

```bash
python3 tools/trace-to-json.py /tmp/trace
```

## Input Format

### Inline (`--input`)

Each `--input` flag provides one function argument as a JSON array. Arguments are positional — first `--input` maps to the first function argument, and so on.

```bash
# For @main(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>)
keren-sim --input='[[1,2,3],[4,5,6]]' --input='[[7,8],[9,10],[11,12]]' model.mlir
```

### File (`--input-file`)

A JSON file containing an array of tensors:

```json
[
  [[1, 2, 3], [4, 5, 6]],
  [[7, 8], [9, 10], [11, 12]]
]
```

Supported element types: `i8`, `i16`, `i32`, `i64`, `f32`, `f64`.

## Output Format

### Default (text)

```
Result 0: tensor<2x2xi32> {
  [
    [11, 22],
    [33, 44]
  ]
}
```

### JSON (`--json`)

```json
{
  "result_0": {
    "type": "tensor<2x2xi32>",
    "value": "tensor<2x2xi32> {\n  [\n    [11, 22],\n    [33, 44]\n  ]\n}"
  }
}
```

### Trace (`--trace=<dir>`)

Creates:
- `<dir>/index.csv` — maps probe IDs to dtype and `.npy` file paths
- `<dir>/probeN.npy` — NumPy binary files with per-op tensor outputs

## Integration with Visualizer

The visualizer's backend (`server.py`) invokes `keren-sim` automatically. See [VISUALIZER.md](VISUALIZER.md) for details.
