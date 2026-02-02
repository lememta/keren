// RUN: %keren-sim %s | FileCheck %s

// CHECK: Result 0:
func.func @main(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %c = stablehlo.add %a, %b : tensor<2x2xi32>
  return %c : tensor<2x2xi32>
}
