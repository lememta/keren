# Keren — ML Graph Compiler Simulation Framework

A C++ toolkit for exploring ML compiler concepts. Includes:

1. **Interpreter** (`keren-sim`): Parses and interprets StableHLO `.mlir` files using the reference interpreter
2. **Compiler** (`keren-compile`): Op-by-op lowering from StableHLO to Linalg with elementwise fusion

## Building

### Prerequisites

- **macOS**: Xcode Command Line Tools, Homebrew
- **Linux**: GCC/Clang, Ninja, CMake

```bash
# macOS
brew install ninja cmake zstd

# Ubuntu/Debian
sudo apt install build-essential ninja-build cmake libzstd-dev
```

### Quick Build (if you have compatible LLVM/MLIR)

If you already have LLVM/MLIR and StableHLO installations:

```bash
cmake -B build -G Ninja \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DSTABLEHLO_SRC_DIR=/path/to/stablehlo \
  -DSTABLEHLO_BUILD_DIR=/path/to/stablehlo/build
cmake --build build
```

### Full Build from Source (macOS)

Keren requires specific LLVM/MLIR and StableHLO versions for API compatibility. Here's how to build everything from source:

#### 1. Build Compatible LLVM/MLIR

```bash
# Clone LLVM at the exact commit StableHLO expects
git clone --depth=1 https://github.com/llvm/llvm-project.git /tmp/llvm-compatible
cd /tmp/llvm-compatible
git fetch origin 01e6245af481dac4604e8a25be6bec0dbe36f99d
git checkout 01e6245af481dac4604e8a25be6bec0dbe36f99d

# Configure and build (15-30 minutes)
mkdir -p /tmp/llvm-build && cd /tmp/llvm-build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  /tmp/llvm-compatible/llvm

ninja mlir-libraries
```

#### 2. Build StableHLO

```bash
# Clone StableHLO
git clone https://github.com/openxla/stablehlo.git /tmp/stablehlo
cd /tmp/stablehlo && git checkout v1.13.8

# Configure and build against compatible LLVM
mkdir -p build && cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DMLIR_DIR=/tmp/llvm-build/lib/cmake/mlir

cmake --build . --parallel 4
```

#### 3. Build Keren

```bash
# Clone and build keren
cd /path/to/keren
npm install  # For visualizer dependencies

cmake -B build -G Ninja \
  -DMLIR_DIR=/tmp/llvm-build/lib/cmake/mlir \
  -DSTABLEHLO_SRC_DIR=/tmp/stablehlo \
  -DSTABLEHLO_BUILD_DIR=/tmp/stablehlo/build

cmake --build build
```

#### 4. Verify Build

```bash
# Test executables
./build/tools/keren-sim --help
./build/tools/keren-compile --help

# Check file sizes (should be ~57MB and ~75MB respectively)
ls -lh build/tools/keren-*
```

### Build Notes

- **macOS**: GNU linker flags (`--start-group`/`--end-group`) are automatically disabled for Apple's linker
- **API Compatibility**: Uses exact LLVM commit `01e6245af481` to ensure StableHLO API compatibility
- **Build Time**: LLVM (~20-30 min), StableHLO (~5 min), Keren (~2 min) on modern hardware
- **Dependencies**: Final executables are statically linked and portable within the same OS

### Troubleshooting

**"No matching function for call to 'create'"**
- Ensure you're using the exact LLVM commit specified above

**"unknown options: --start-group --end-group"**
- This is automatically handled on macOS; if you see this, clean and reconfigure

**Missing StableHLO headers**
- Verify `STABLEHLO_SRC_DIR` and `STABLEHLO_BUILD_DIR` point to correct paths

## Tools

### keren-sim — StableHLO Interpreter

```bash
# Interpret a StableHLO module
build/tools/keren-sim input.mlir

# With input values
build/tools/keren-sim --input='[[1,2],[3,4]]' --input='[[5,6],[7,8]]' input.mlir

# JSON output
build/tools/keren-sim --json input.mlir

# Trace intermediate values
build/tools/keren-sim --trace=/tmp/trace input.mlir
```

### keren-compile — StableHLO to Linalg Compiler

```bash
# Op-by-op lowering
build/tools/keren-compile --lower-ops input.mlir

# With elementwise fusion
build/tools/keren-compile --lower-ops --fuse input.mlir

# Verbose mode (shows each compilation stage)
build/tools/keren-compile --lower-ops --fuse --verbose input.mlir

# Output to file
build/tools/keren-compile --lower-ops --fuse -o output.mlir input.mlir
```

**Supported Operations:**

| Category | StableHLO Ops | Linalg Lowering |
|----------|---------------|-----------------|
| Elementwise | add, mul, sub, div, exp, sqrt, tanh, log, negate | `linalg.generic` |
| Contraction | dot_general | `linalg.matmul`, `linalg.batch_matmul` |
| Reduction | reduce (sum, max, min) | `linalg.reduce` |
| Shape | broadcast_in_dim | `linalg.broadcast` |

**Fusion Example:**

```mlir
// Input
%0 = stablehlo.add %a, %b : tensor<16x64xf32>
%1 = stablehlo.multiply %0, %c : tensor<16x64xf32>

// Output (fused)
%1 = linalg.generic {...} ins(%a, %b, %c) {
  %t = arith.addf %a_elem, %b_elem
  %r = arith.mulf %t, %c_elem
  linalg.yield %r
}
```

See [docs/COMPILER_README.md](docs/COMPILER_README.md) for full documentation.

## Testing

```bash
# All tests
cmake --build build --target check-keren

# Compiler tests only
build/tools/keren-compile --lower-ops test/compiler/elementwise.mlir
build/tools/keren-compile --lower-ops --fuse test/compiler/fusion.mlir
```

## Architecture

```
keren/
├── tools/
│   ├── keren-sim.cpp        # Interpreter CLI
│   └── keren-compile.cpp    # Compiler CLI
├── lib/
│   ├── Interpreter.cpp      # Module evaluation driver
│   ├── Compiler.cpp         # Compilation pipeline
│   ├── Fusion.cpp           # Elementwise fusion pass
│   └── Lowerings/           # Op-by-op lowerings
├── include/keren/           # Headers
├── contracts/
│   └── compiler.sage        # SAGE formal contracts
├── visualizer/              # Web-based graph viewer
└── examples/
    ├── gpt2.mlir            # GPT-2 model
    └── mini_attention.mlir  # Attention block example
```

**Key Components:**

- **Interpreter**: Parses MLIR, dispatches to dialect handlers, uses StableHLO reference interpreter
- **Compiler**: Op-by-op lowering with pattern rewriting, optional fusion pass
- **OpLowering**: Base class for individual op conversions (elementwise, contraction, reduction)
- **FusionPass**: Identifies and merges consecutive elementwise `linalg.generic` operations
- **Visualizer**: Web UI for exploring computation graphs

## Documentation

- [Simulator Guide](docs/SIMULATOR.md)
- [Compiler Guide](docs/COMPILER_README.md)
- [Visualizer Guide](docs/VISUALIZER.md)
- [Architecture](docs/COMPILER.md)
- [SAGE Contracts](contracts/compiler.sage)

## License

Apache-2.0
