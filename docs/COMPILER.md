# Keren Op-by-Op Compiler

A pedagogical StableHLO → Linalg compiler that lowers operations one at a time, with support for operation fusion.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        StableHLO Module                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │  dot    │→│ multiply│→│  add    │→│ reduce  │              │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Op-by-Op Lowering Pass                        │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ OpLoweringBase   │  │ OpLoweringPattern│                     │
│  │  - match()       │  │  - DotLowering   │                     │
│  │  - rewrite()     │  │  - AddLowering   │                     │
│  └──────────────────┘  │  - MulLowering   │                     │
│                        │  - ReduceLowering│                     │
│                        └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Fusion Analysis                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Identify fusable op chains:                                  ││
│  │   elementwise → elementwise (producer-consumer)              ││
│  │   broadcast → elementwise                                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Linalg Module                             │
│  ┌─────────────┐  ┌─────────────────────────────────┐          │
│  │linalg.matmul│→│    linalg.generic (fused)       │          │
│  └─────────────┘  │  {mul + add in single kernel}   │          │
│                   └─────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Pedagogical Clarity**: Each op lowering is self-contained and documented
2. **One-to-One Mapping**: Clear correspondence between StableHLO ops and Linalg ops
3. **Composable Fusion**: Fusion is a separate pass that combines compatible Linalg ops
4. **SAGE Contracts**: All transformations have formal pre/post conditions

## Supported Operations

### Elementwise Operations
| StableHLO Op | Linalg Lowering | Fusable |
|--------------|-----------------|---------|
| `add` | `linalg.generic` | ✓ |
| `multiply` | `linalg.generic` | ✓ |
| `subtract` | `linalg.generic` | ✓ |
| `divide` | `linalg.generic` | ✓ |
| `negate` | `linalg.generic` | ✓ |
| `exponential` | `linalg.generic` | ✓ |
| `sqrt` | `linalg.generic` | ✓ |

### Contraction Operations
| StableHLO Op | Linalg Lowering | Fusable |
|--------------|-----------------|---------|
| `dot_general` | `linalg.matmul` / `linalg.batch_matmul` | ✗ (producer only) |

### Reduction Operations
| StableHLO Op | Linalg Lowering | Fusable |
|--------------|-----------------|---------|
| `reduce` | `linalg.reduce` | ✗ (consumer only) |

### Shape Operations
| StableHLO Op | Linalg Lowering | Fusable |
|--------------|-----------------|---------|
| `broadcast_in_dim` | `linalg.broadcast` | ✓ (into consumer) |
| `reshape` | `tensor.collapse_shape` / `tensor.expand_shape` | - |
| `transpose` | `linalg.transpose` | - |

## Fusion Strategy

The compiler implements **vertical fusion** (producer-consumer fusion) for elementwise operations:

```
Before Fusion:
  %0 = linalg.generic {add}(%a, %b)
  %1 = linalg.generic {mul}(%0, %c)

After Fusion:
  %1 = linalg.generic {
    ^bb0(%a, %b, %c, %out):
      %t = arith.addf %a, %b
      %r = arith.mulf %t, %c
      linalg.yield %r
  }(%a, %b, %c)
```

### Fusion Rules

1. **Element-wise chains**: Adjacent elementwise ops with matching iteration domains fuse
2. **Broadcast absorption**: `broadcast_in_dim` fuses into consuming elementwise ops
3. **No fusion across**:
   - Reductions (synchronization boundary)
   - Contractions (complex iteration domain)
   - Ops with multiple consumers (would duplicate work)

## Usage

```bash
# Op-by-op lowering only
keren-compile --lower-ops input.mlir -o lowered.mlir

# Op-by-op lowering + fusion
keren-compile --lower-ops --fuse input.mlir -o fused.mlir

# Verbose mode (shows each lowering step)
keren-compile --lower-ops --verbose input.mlir
```

## Example: Mini-Attention Block

Input (StableHLO):
```mlir
func.func @attention(%x: tensor<16x64xf32>, %w: tensor<64x64xf32>) -> tensor<16x64xf32> {
  %0 = stablehlo.dot_general %x, %w : (tensor<16x64xf32>, tensor<64x64xf32>) -> tensor<16x64xf32>
  %1 = stablehlo.multiply %0, %0 : tensor<16x64xf32>
  %c = stablehlo.constant dense<64.0> : tensor<f32>
  %2 = stablehlo.broadcast_in_dim %c : (tensor<f32>) -> tensor<16x64xf32>
  %3 = stablehlo.divide %1, %2 : tensor<16x64xf32>
  return %3 : tensor<16x64xf32>
}
```

Output (Linalg, after fusion):
```mlir
func.func @attention(%x: tensor<16x64xf32>, %w: tensor<64x64xf32>) -> tensor<16x64xf32> {
  // Matmul cannot fuse
  %0 = linalg.matmul ins(%x, %w) outs(%init) -> tensor<16x64xf32>
  
  // Fused: square + divide_by_64
  %c = arith.constant 64.0 : f32
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%0) outs(%init) {
    ^bb0(%in: f32, %out: f32):
      %sq = arith.mulf %in, %in : f32    // multiply
      %div = arith.divf %sq, %c : f32    // divide by broadcast constant
      linalg.yield %div : f32
  } -> tensor<16x64xf32>
  
  return %1 : tensor<16x64xf32>
}
```

## File Structure

```
keren/
├── include/keren/
│   ├── Compiler.h           # Main compiler driver
│   ├── OpLowering.h         # Op lowering base class & registry
│   ├── Fusion.h             # Fusion analysis & pass
│   └── Lowerings/
│       ├── ElementwiseOps.h # add, mul, sub, div, exp, sqrt
│       ├── ContractionOps.h # dot_general
│       └── ReductionOps.h   # reduce
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
└── test/
    └── compiler/
        ├── elementwise.mlir
        ├── fusion.mlir
        └── matmul.mlir
```

## References

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- Blog: [From JAX to VLIW](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
