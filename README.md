# Keren — ML Graph Compiler Simulation Framework

A C++ toolkit for exploring ML compiler concepts. Includes:

1. **Interpreter** (`keren-sim`): Parses and interprets StableHLO `.mlir` files using the reference interpreter
2. **Compiler** (`keren-compile`): Op-by-op lowering from StableHLO to Linalg with elementwise fusion

## Building

Requires pre-built LLVM/MLIR and StableHLO installations.

```bash
cmake -B build -G Ninja \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DStableHLO_DIR=/path/to/stablehlo/lib/cmake/stablehlo
cmake --build build
```

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
