// RUN: %keren-sim %s | FileCheck %s

// CHECK: Result 0:
func.func @main() -> tensor<3xf32> {
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %b = stablehlo.constant dense<[4.0, 5.0, 6.0]> : tensor<3xf32>
  %c = stablehlo.multiply %a, %b : tensor<3xf32>
  return %c : tensor<3xf32>
}
