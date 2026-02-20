// RUN: keren-compile --lower-ops %s | FileCheck %s

// Test dot_general lowering to linalg.matmul

// Standard 2D matmul: (M, K) x (K, N) -> (M, N)
// CHECK-LABEL: func.func @test_matmul_2d
// CHECK: linalg.matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<16x32xf32>, tensor<32x64xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<16x64xf32>)
func.func @test_matmul_2d(%a: tensor<16x32xf32>, %b: tensor<32x64xf32>) -> tensor<16x64xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0]
       : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
  return %0 : tensor<16x64xf32>
}

// Batch matmul: (B, M, K) x (B, K, N) -> (B, M, N)
// CHECK-LABEL: func.func @test_batch_matmul
// CHECK: linalg.batch_matmul
func.func @test_batch_matmul(%a: tensor<4x16x32xf32>, %b: tensor<4x32x64xf32>) -> tensor<4x16x64xf32> {
  %0 = stablehlo.dot_general %a, %b, 
       batching_dims = [0] x [0],
       contracting_dims = [2] x [1]
       : (tensor<4x16x32xf32>, tensor<4x32x64xf32>) -> tensor<4x16x64xf32>
  return %0 : tensor<4x16x64xf32>
}

// Integer matmul
// CHECK-LABEL: func.func @test_matmul_int
// CHECK: linalg.matmul
func.func @test_matmul_int(%a: tensor<8x16xi32>, %b: tensor<16x8xi32>) -> tensor<8x8xi32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0]
       : (tensor<8x16xi32>, tensor<16x8xi32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// Matmul followed by bias add (common pattern)
// CHECK-LABEL: func.func @test_matmul_bias
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.addf
func.func @test_matmul_bias(%x: tensor<16x64xf32>, %w: tensor<64x32xf32>, 
                            %bias: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
       : (tensor<16x64xf32>, tensor<64x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.add %0, %bias : tensor<16x32xf32>
  return %1 : tensor<16x32xf32>
}
