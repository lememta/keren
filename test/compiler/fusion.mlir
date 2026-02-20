// RUN: keren-compile --lower-ops --fuse %s | FileCheck %s

// Test elementwise operation fusion

// This test demonstrates vertical fusion of elementwise operations.
// The add and multiply operations should be fused into a single linalg.generic.

// CHECK-LABEL: func.func @test_fusion_add_mul
// Before fusion: 2 linalg.generic ops
// After fusion: 1 linalg.generic op with both add and mul in body
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: linalg.yield
// CHECK-NOT: linalg.generic
func.func @test_fusion_add_mul(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, 
                               %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  // %0 = a + b
  %0 = stablehlo.add %a, %b : tensor<16x16xf32>
  // %1 = %0 * c = (a + b) * c
  %1 = stablehlo.multiply %0, %c : tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}

// Test chain of 3 operations
// CHECK-LABEL: func.func @test_fusion_chain
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: linalg.yield
func.func @test_fusion_chain(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>,
                             %c: tensor<8x8xf32>, %d: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %a, %b : tensor<8x8xf32>
  %1 = stablehlo.multiply %0, %c : tensor<8x8xf32>
  %2 = stablehlo.subtract %1, %d : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// Test that matmul is NOT fused with elementwise (fusion barrier)
// CHECK-LABEL: func.func @test_no_fusion_matmul
// CHECK: linalg.matmul
// CHECK: linalg.generic
func.func @test_no_fusion_matmul(%a: tensor<16x32xf32>, %b: tensor<32x16xf32>,
                                  %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] 
       : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
  %1 = stablehlo.add %0, %c : tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}

// Test exp + tanh chain (math ops)
// CHECK-LABEL: func.func @test_fusion_math_ops
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: math.tanh
func.func @test_fusion_math_ops(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.exponential %a : tensor<4x4xf32>
  %1 = stablehlo.tanh %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
