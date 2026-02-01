// RUN: %keren-sim %s | FileCheck %s

// CHECK: Result 0:
func.func @main() -> tensor<2x2xi32> {
  %a = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %b = stablehlo.constant dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>
  %c = stablehlo.add %a, %b : tensor<2x2xi32>
  return %c : tensor<2x2xi32>
}
