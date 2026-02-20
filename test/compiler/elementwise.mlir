// RUN: keren-compile --lower-ops %s | FileCheck %s

// Test basic elementwise operations lowering

// CHECK-LABEL: func.func @test_add
// CHECK: linalg.generic
// CHECK: arith.addf
func.func @test_add(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.add %a, %b : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_multiply
// CHECK: linalg.generic
// CHECK: arith.mulf
func.func @test_multiply(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.multiply %a, %b : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_exp
// CHECK: linalg.generic
// CHECK: math.exp
func.func @test_exp(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.exponential %a : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_sqrt
// CHECK: linalg.generic
// CHECK: math.sqrt
func.func @test_sqrt(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.sqrt %a : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_divide
// CHECK: linalg.generic
// CHECK: arith.divf
func.func @test_divide(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.divide %a, %b : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_negate
// CHECK: linalg.generic
// CHECK: arith.negf
func.func @test_negate(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.negate %a : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
