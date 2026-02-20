@mod keren_compiler
"StableHLO to Linalg op-by-op compiler with fusion support"

---

@spec OpLowering
"Contract for individual operation lowerings"

@type TensorType = {
  shape: [Int],
  elementType: ElementType,
  encoding: Str?
}

@type ElementType = F16 | F32 | F64 | I8 | I16 | I32 | I64 | BF16

@type Operation = {
  name: Str,
  operands: [Value],
  results: [Value],
  attributes: Map<Str, Any>
}

@type Value = {
  type: TensorType,
  definingOp: Operation?
}

@state
  inputOp: Operation
  outputOps: [Operation]

---

@op lower_elementwise_binary(op: Operation) -> [Operation]
"Lower a binary elementwise StableHLO op to linalg.generic"

@req op.name ∈ {"stablehlo.add", "stablehlo.multiply", "stablehlo.subtract", "stablehlo.divide"}
@req |op.operands| = 2
@req |op.results| = 1
@req op.operands[0].type.shape = op.operands[1].type.shape
  !! "Implicit broadcast must be materialized before lowering"

@ens |result| = 1
@ens result[0].name = "linalg.generic"
@ens result[0].results[0].type = op.results[0].type
  !! "Type preservation: output shape and dtype unchanged"

@ens ∀ i ∈ result[0].indexing_maps: is_identity_map(i)
  !! "Elementwise ops use identity indexing maps"

@ens ∀ d ∈ result[0].iterator_types: d = "parallel"
  !! "All dimensions are parallel (no reduction)"

---

@op lower_dot_general(op: Operation) -> [Operation]
"Lower stablehlo.dot_general to linalg.matmul or linalg.batch_matmul"

@req op.name = "stablehlo.dot_general"
@req |op.operands| = 2
@req |op.results| = 1

"Extract contraction dimensions from dot_dimension_numbers"
let lhs_contracting = op.attributes["dot_dimension_numbers"].lhs_contracting_dimensions
let rhs_contracting = op.attributes["dot_dimension_numbers"].rhs_contracting_dimensions
let lhs_batch = op.attributes["dot_dimension_numbers"].lhs_batching_dimensions
let rhs_batch = op.attributes["dot_dimension_numbers"].rhs_batching_dimensions

@req |lhs_contracting| = |rhs_contracting|
  !! "Contracting dimensions must match"
@req |lhs_batch| = |rhs_batch|
  !! "Batch dimensions must match"

@ens |result| >= 1
@ens result[0].name ∈ {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}
@ens result[0].results[0].type = op.results[0].type

---

@op lower_reduce(op: Operation) -> [Operation]
"Lower stablehlo.reduce to linalg.reduce"

@req op.name = "stablehlo.reduce"
@req |op.operands| >= 2
  !! "At least one input tensor and one init value"
@req |op.results| >= 1

let reduction_dims = op.attributes["dimensions"]
let input_rank = |op.operands[0].type.shape|
let output_rank = |op.results[0].type.shape|

@req output_rank = input_rank - |reduction_dims|
  !! "Output rank reduced by number of reduction dimensions"

@ens |result| = 1
@ens result[0].name = "linalg.reduce"
@ens result[0].results[0].type = op.results[0].type

---

@op lower_broadcast_in_dim(op: Operation) -> [Operation]
"Lower stablehlo.broadcast_in_dim to linalg.broadcast"

@req op.name = "stablehlo.broadcast_in_dim"
@req |op.operands| = 1
@req |op.results| = 1

let broadcast_dims = op.attributes["broadcast_dimensions"]
let input_shape = op.operands[0].type.shape
let output_shape = op.results[0].type.shape

@req |broadcast_dims| = |input_shape|
@req ∀ i ∈ 0..|broadcast_dims|:
  input_shape[i] = 1 ∨ input_shape[i] = output_shape[broadcast_dims[i]]
  !! "Broadcast dimensions must be compatible"

@ens |result| = 1
@ens result[0].name = "linalg.broadcast"
@ens result[0].results[0].type = op.results[0].type

---

@spec Fusion
"Contract for operation fusion"

@type FusionCandidate = {
  producer: Operation,
  consumer: Operation,
  fusionKind: FusionKind
}

@type FusionKind = ElementwiseChain | BroadcastAbsorption | None

---

@op can_fuse(producer: Operation, consumer: Operation) -> Bool
"Determine if two operations can be fused"

@req producer.results ∩ consumer.operands ≠ ∅
  !! "Producer must feed into consumer"

"Fusion rules"
let is_elementwise = λ op: op.name ∈ 
  {"linalg.generic"} ∧ ∀ d ∈ op.iterator_types: d = "parallel"

@ens result = True ⟹ (
  (is_elementwise(producer) ∧ is_elementwise(consumer))
  ∨ (producer.name = "linalg.broadcast" ∧ is_elementwise(consumer))
)
  !! "Only elementwise chains and broadcast absorption are supported"

@ens result = True ⟹ |users(producer.results[0])| = 1
  !! "No fusion if producer has multiple consumers (would duplicate work)"

@ens result = False ⟹ (
  producer.name ∈ {"linalg.matmul", "linalg.batch_matmul", "linalg.reduce"}
  ∨ consumer.name ∈ {"linalg.reduce"}
)
  !! "Contractions and reductions are fusion barriers"

---

@op fuse_elementwise(ops: [Operation]) -> Operation
"Fuse a chain of elementwise operations into a single linalg.generic"

@req |ops| >= 2
@req ∀ op ∈ ops: is_elementwise(op)
@req ∀ i ∈ 1..|ops|: ops[i-1].results[0] ∈ ops[i].operands
  !! "Operations form a producer-consumer chain"

let combined_inputs = external_inputs(ops)
let final_output = ops[-1].results[0]

@ens result.name = "linalg.generic"
@ens result.results[0].type = final_output.type
@ens result.operands = combined_inputs
  !! "Fused op takes all external inputs"
@ens result.body contains ∀ op ∈ ops: op.body
  !! "Fused body contains all original computations"

---

@spec CompilerPipeline
"End-to-end compilation contract"

@type Module = {
  functions: [Function],
  globals: [Value]
}

@type Function = {
  name: Str,
  arguments: [Value],
  results: [TensorType],
  body: [Operation]
}

---

@op compile(input: Module, options: CompilerOptions) -> Module
"Compile StableHLO module to Linalg"

@req ∀ op ∈ all_ops(input): op.dialect ∈ {"stablehlo", "func", "arith"}
  !! "Input must be valid StableHLO"

@ens ∀ op ∈ all_ops(result): op.dialect ∈ {"linalg", "tensor", "arith", "func"}
  !! "Output contains only Linalg and supporting dialects"

@ens ∀ f ∈ input.functions:
  semantics(f) = semantics(result.functions[f.name])
  !! "Semantic preservation: computation produces same results"

@ens options.fuse = True ⟹
  |all_ops(result)| <= |all_ops(lowered_without_fusion)|
  !! "Fusion reduces or maintains operation count"

---

@invariant type_preservation
"All lowerings preserve tensor types"
∀ lowering ∈ {lower_elementwise_binary, lower_dot_general, lower_reduce}:
  ∀ input_op, output_ops = lowering(input_op):
    input_op.results[0].type = output_ops[-1].results[0].type

@invariant semantic_equivalence  
"Lowered operations compute the same mathematical function"
∀ lowering, input_op, output_ops = lowering(input_op):
  ∀ inputs: eval(input_op, inputs) = eval(compose(output_ops), inputs)

@invariant fusion_correctness
"Fused operations compute the same result as unfused chain"
∀ ops, fused_op = fuse_elementwise(ops):
  ∀ inputs: eval(compose(ops), inputs) = eval(fused_op, inputs)

---

!! "Implementation notes"

!! "Elementwise lowering uses linalg.generic with:
    - Identity indexing maps (affine_map<(d0, d1, ...) -> (d0, d1, ...)>)
    - All parallel iterator types
    - Body contains single arith operation"

!! "Fusion merges multiple linalg.generic ops by:
    1. Combining indexing maps
    2. Concatenating input operands (deduplicating shared inputs)
    3. Inlining producer bodies into consumer body
    4. Eliminating intermediate tensor allocations"

!! "Matmul lowering handles dot_general dimension permutations:
    - Standard matmul: lhs_contracting=[1], rhs_contracting=[0]
    - Batch matmul adds batch dimensions to both sides
    - Non-standard layouts require transpose insertion"
