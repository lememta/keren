# Keren — ML Graph Compiler Simulation Framework

A C++ CLI tool that parses StableHLO `.mlir` files and interprets them to produce output tensors. Uses MLIR/StableHLO libraries for parsing and wraps StableHLO's reference interpreter (`evalModule`) for execution.

## Building

Requires pre-built LLVM/MLIR and StableHLO installations.

```bash
cmake -B build -G Ninja \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DStableHLO_DIR=/path/to/stablehlo/lib/cmake/stablehlo
cmake --build build
```

## Usage

```bash
# Interpret a StableHLO module
build/tools/keren-sim input.mlir

# Lower to Linalg first (prints lowered IR)
build/tools/keren-sim --lower-to-linalg input.mlir

# Specify entry function
build/tools/keren-sim --entry=my_func input.mlir
```

## Testing

```bash
cmake --build build --target check-keren
```

## Architecture

- **Interpreter**: Driver that parses MLIR, manages dialect handlers, and orchestrates evaluation
- **DialectHandler**: Abstract base class for dialect-specific interpreters (`canHandle` + `evaluate`)
- **StableHLOHandler**: Delegates to `stablehlo::evalModule` for whole-module interpretation
- **LinalgHandler**: Stub for future Linalg op interpretation
- **Pipeline**: Wraps MLIR pass pipelines (StableHLO -> Linalg lowering)
- **TensorValue**: Wrapper around `stablehlo::Tensor`, extensible for memref semantics

## License

Apache-2.0
