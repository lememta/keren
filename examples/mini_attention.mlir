// Mini-Attention Block Example
//
// This example demonstrates the Keren compiler's op-by-op lowering and fusion
// on a simplified attention-like computation:
//   matmul → square → divide → reduce_max → subtract → exp → reduce_sum → divide
//
// Usage:
//   # Op-by-op lowering only
//   keren-compile --lower-ops examples/mini_attention.mlir
//
//   # With elementwise fusion
//   keren-compile --lower-ops --fuse examples/mini_attention.mlir
//
//   # Verbose mode to see each stage
//   keren-compile --lower-ops --fuse --verbose examples/mini_attention.mlir

// Main entry point: mini_attention block
// Input:  x[16, 64], w1[64, 64], w2[64, 32]
// Output: out[16, 32]
func.func @mini_attention(%x: tensor<16x64xf32>, 
                          %w1: tensor<64x64xf32>,
                          %w2: tensor<64x32xf32>) -> tensor<16x32xf32> {
  
  // Stage 1: Linear projection (like Q @ K^T)
  // h = x @ w1  [16, 64] @ [64, 64] -> [16, 64]
  %h = stablehlo.dot_general %x, %w1, contracting_dims = [1] x [0]
       : (tensor<16x64xf32>, tensor<64x64xf32>) -> tensor<16x64xf32>
  
  // Stage 2: RMS Normalization
  // rms = sqrt(mean(h^2, axis=-1, keepdims=True) + eps)
  // h_norm = h / rms
  
  // h_squared = h * h
  %h_squared = stablehlo.multiply %h, %h : tensor<16x64xf32>
  
  // sum_squared = reduce_sum(h_squared, dim=1)
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %sum_squared = stablehlo.reduce(%h_squared init: %zero) across dimensions = [1] 
    : (tensor<16x64xf32>, tensor<f32>) -> tensor<16xf32>
    reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
      %add = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }
  
  // mean_squared = sum_squared / 64.0
  %dims_f = stablehlo.constant dense<64.0> : tensor<16xf32>
  %mean_squared = stablehlo.divide %sum_squared, %dims_f : tensor<16xf32>
  
  // eps = 1e-6 for numerical stability
  %eps = stablehlo.constant dense<1.0e-6> : tensor<16xf32>
  %mean_plus_eps = stablehlo.add %mean_squared, %eps : tensor<16xf32>
  
  // rms = sqrt(mean_squared + eps)
  %rms = stablehlo.sqrt %mean_plus_eps : tensor<16xf32>
  
  // Broadcast rms back to [16, 64] shape
  %rms_broadcast = stablehlo.broadcast_in_dim %rms, dims = [0] 
    : (tensor<16xf32>) -> tensor<16x64xf32>
  
  // h_norm = h / rms
  %h_norm = stablehlo.divide %h, %rms_broadcast : tensor<16x64xf32>
  
  // Stage 3: Softmax (row-wise, numerically stable)
  // h_max = max(h_norm, axis=-1, keepdims=True)
  // exp_h = exp(h_norm - h_max)
  // h_soft = exp_h / sum(exp_h, axis=-1, keepdims=True)
  
  // Find max per row for numerical stability
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
  %h_max = stablehlo.reduce(%h_norm init: %neg_inf) across dimensions = [1]
    : (tensor<16x64xf32>, tensor<f32>) -> tensor<16xf32>
    reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
      %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %max : tensor<f32>
    }
  
  // Broadcast max back
  %h_max_broadcast = stablehlo.broadcast_in_dim %h_max, dims = [0]
    : (tensor<16xf32>) -> tensor<16x64xf32>
  
  // Subtract max for stability
  %h_shifted = stablehlo.subtract %h_norm, %h_max_broadcast : tensor<16x64xf32>
  
  // Exponential
  %exp_h = stablehlo.exponential %h_shifted : tensor<16x64xf32>
  
  // Sum of exponentials
  %exp_sum = stablehlo.reduce(%exp_h init: %zero) across dimensions = [1]
    : (tensor<16x64xf32>, tensor<f32>) -> tensor<16xf32>
    reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
      %add = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }
  
  // Broadcast sum back
  %exp_sum_broadcast = stablehlo.broadcast_in_dim %exp_sum, dims = [0]
    : (tensor<16xf32>) -> tensor<16x64xf32>
  
  // Normalize: softmax output
  %h_soft = stablehlo.divide %exp_h, %exp_sum_broadcast : tensor<16x64xf32>
  
  // Stage 4: Output projection (like attention @ V)
  // out = h_soft @ w2  [16, 64] @ [64, 32] -> [16, 32]
  %out = stablehlo.dot_general %h_soft, %w2, contracting_dims = [1] x [0]
         : (tensor<16x64xf32>, tensor<64x32xf32>) -> tensor<16x32xf32>
  
  return %out : tensor<16x32xf32>
}

// Simpler test case: just elementwise ops that should fuse
func.func @test_fusion(%a: tensor<16x64xf32>, %b: tensor<16x64xf32>) -> tensor<16x64xf32> {
  // These 3 ops should fuse into a single linalg.generic
  %0 = stablehlo.add %a, %b : tensor<16x64xf32>
  %1 = stablehlo.multiply %0, %0 : tensor<16x64xf32>  // square the sum
  %2 = stablehlo.tanh %1 : tensor<16x64xf32>
  return %2 : tensor<16x64xf32>
}
