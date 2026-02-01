// RUN: %keren-sim %s | FileCheck %s

// CHECK: Result 0:
func.func @main() -> tensor<2x2xf32> {
  %a = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]> : tensor<3x2xf32>
  %c = stablehlo.dot_general %a, %b,
    contracting_dims = [1] x [0] :
    (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}
