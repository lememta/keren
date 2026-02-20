# Keren Compiler: StableHLO to Linalg with Op-by-Op Lowering

This document describes the Keren compiler extension that lowers StableHLO operations to the Linalg dialect one operation at a time, with optional elementwise fusion.

## Motivation

Inspired by [From JAX to VLIW: Tracing a Computation Through the TPU Compiler Stack](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation), this compiler demonstrates:

1. **Op-by-Op Lowering**: Each StableHLO operation is individually converted to its Linalg equivalent
2. **Operation Fusion**: Consecutive elementwise operations are fused to reduce memory bandwidth
3. **Pedagogical Clarity**: The code is structured to be readable and educational

## Quick Start

```bash
# Build
cd keren/build
cmake .. -DLLVM_DIR=/path/to/llvm -DStablehlo_DIR=/path/to/stablehlo
cmake --build . --target keren-compile

# Run with op-by-op lowering
./tools/keren-compile --lower-ops input.mlir

# Run with fusion
./tools/keren-compile --lower-ops --fuse input.mlir

# Verbose mode
./tools/keren-compile --lower-ops --fuse --verbose input.mlir
```

## Supported Operations

### Elementwise Binary Operations
| StableHLO | Linalg | Arithmetic Op |
|-----------|--------|---------------|
| `stablehlo.add` | `linalg.generic` | `arith.addf` / `arith.addi` |
| `stablehlo.multiply` | `linalg.generic` | `arith.mulf` / `arith.muli` |
| `stablehlo.subtract` | `linalg.generic` | `arith.subf` / `arith.subi` |
| `stablehlo.divide` | `linalg.generic` | `arith.divf` / `arith.divsi` |
| `stablehlo.maximum` | `linalg.generic` | `arith.maximumf` / `arith.maxsi` |

### Elementwise Unary Operations
| StableHLO | Linalg | Math Op |
|-----------|--------|---------|
| `stablehlo.negate` | `linalg.generic` | `arith.negf` |
| `stablehlo.exponential` | `linalg.generic` | `math.exp` |
| `stablehlo.sqrt` | `linalg.generic` | `math.sqrt` |
| `stablehlo.rsqrt` | `linalg.generic` | `math.rsqrt` |
| `stablehlo.log` | `linalg.generic` | `math.log` |
| `stablehlo.tanh` | `linalg.generic` | `math.tanh` |

### Contraction Operations
| StableHLO | Linalg |
|-----------|--------|
| `stablehlo.dot_general` (2D) | `linalg.matmul` |
| `stablehlo.dot_general` (batched) | `linalg.batch_matmul` |
| `stablehlo.dot_general` (general) | `linalg.generic` with reduction |

### Reduction Operations
| StableHLO | Linalg |
|-----------|--------|
| `stablehlo.reduce` (sum) | `linalg.reduce` |
| `stablehlo.reduce` (max) | `linalg.reduce` |
| `stablehlo.reduce` (min) | `linalg.reduce` |
| `stablehlo.reduce` (prod) | `linalg.reduce` |

### Shape Operations
| StableHLO | Linalg |
|-----------|--------|
| `stablehlo.broadcast_in_dim` | `linalg.broadcast` |

## Fusion Strategy

The compiler implements **vertical fusion** for elementwise operations:

### What Gets Fused
- Chains of elementwise operations (add, mul, sub, div, exp, sqrt, etc.)
- Broadcast operations absorbed into consuming elementwise ops

### What Does NOT Fuse
- Operations with multiple consumers (would duplicate computation)
- Reduction operations (synchronization boundary)
- Contraction operations (matmul, batch_matmul)

### Example

```mlir
// Input (StableHLO)
%0 = stablehlo.add %a, %b : tensor<16x64xf32>
%1 = stablehlo.multiply %0, %c : tensor<16x64xf32>

// Output (Linalg, after fusion)
%1 = linalg.generic {
  indexing_maps = [identity, identity, identity, identity],
  iterator_types = ["parallel", "parallel"]
} ins(%a, %b, %c) outs(%init) {
  ^bb0(%a_elem, %b_elem, %c_elem, %out):
    %t = arith.addf %a_elem, %b_elem : f32
    %r = arith.mulf %t, %c_elem : f32
    linalg.yield %r : f32
} -> tensor<16x64xf32>
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     keren-compile CLI                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Compiler.cpp                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Parse MLIR  │→│ Op Lowering │→│   Fusion    │→ Output  │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ElementwiseOps.cpp│  │ContractionOps.cpp│  │ ReductionOps.cpp│
│  AddOpLowering  │  │DotGeneralLowering│  │ReduceOpLowering│
│  MulOpLowering  │  │                 │  │BroadcastLowering│
│  ExpOpLowering  │  │                 │  │                 │
│  ...            │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## SAGE Contracts

The compiler is documented with SAGE (Semi-formal AI-Guided Engineering) contracts in `contracts/compiler.sage`. Key contracts include:

```sage
@op lower_elementwise_binary(op: Operation) -> [Operation]
@req op.operands[0].type.shape = op.operands[1].type.shape
@ens result[0].name = "linalg.generic"
@ens ∀ d ∈ result[0].iterator_types: d = "parallel"
```

```sage
@op can_fuse(producer: Operation, consumer: Operation) -> Bool
@req producer.results ∩ consumer.operands ≠ ∅
@ens result = True ⟹ is_elementwise(producer) ∧ is_elementwise(consumer)
@ens result = True ⟹ |users(producer.results[0])| = 1
```

## Files

```
keren/
├── include/keren/
│   ├── Compiler.h           # Main compiler driver
│   ├── OpLowering.h         # Op lowering base class & registry
│   ├── Fusion.h             # Fusion analysis & pass
│   └── Lowerings/
│       ├── ElementwiseOps.h # add, mul, sub, div, exp, sqrt
│       ├── ContractionOps.h # dot_general
│       └── ReductionOps.h   # reduce, broadcast
├── lib/
│   ├── Compiler.cpp
│   ├── OpLowering.cpp
│   ├── Fusion.cpp
│   └── Lowerings/
│       ├── ElementwiseOps.cpp
│       ├── ContractionOps.cpp
│       └── ReductionOps.cpp
├── tools/
│   └── keren-compile.cpp    # CLI tool
├── contracts/
│   └── compiler.sage        # SAGE contracts
├── docs/
│   ├── COMPILER.md          # Architecture overview
│   └── COMPILER_README.md   # This file
├── test/compiler/
│   ├── elementwise.mlir     # Elementwise lowering tests
│   ├── fusion.mlir          # Fusion tests
│   └── matmul.mlir          # Matmul lowering tests
└── examples/
    └── mini_attention.mlir  # Mini-attention block example
```

## Testing

```bash
# Run all compiler tests
cd build
cmake --build . --target check-keren-compile

# Or run individual tests
./tools/keren-compile --lower-ops ../test/compiler/elementwise.mlir
./tools/keren-compile --lower-ops --fuse ../test/compiler/fusion.mlir
./tools/keren-compile --lower-ops ../test/compiler/matmul.mlir
```

## Future Work

- [ ] Support for more StableHLO operations (scatter, gather, pad, slice)
- [ ] Horizontal fusion (parallel independent ops)
- [ ] Loop tiling for cache optimization
- [ ] Vectorization pass
- [ ] Backend codegen (LLVM, CUDA, etc.)

## References

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [From JAX to VLIW](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
