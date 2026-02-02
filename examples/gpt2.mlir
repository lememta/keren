module @gpt2 {

// ── Token & Position Embedding ──
  func.func @token_embedding(%arg0: tensor<256x128xf32>, %arg1: tensor<1x16xi32>) -> (tensor<1x16x128xf32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<1x16xi1>, tensor<1x16xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<1x16x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<256x128xf32>, tensor<1x16x1xi32>) -> tensor<1x16x128xf32>
    return %6 : tensor<1x16x128xf32>
  }

  func.func @position_embedding(%arg0: tensor<16x128xf32>, %arg1: tensor<16xi32>) -> (tensor<16x128xf32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16xi1>, tensor<16xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<16x128xf32>, tensor<16x1xi32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }

// ── Layer Normalization ──
  func.func @layer_norm(%arg0: tensor<1x16x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<1x16x128xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_0 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %3 = stablehlo.divide %1, %2 : tensor<1x16x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x16x128xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x16x128xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_2 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %10 = stablehlo.divide %8, %9 : tensor<1x16x1xf32>
    %11 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %12 = stablehlo.subtract %arg0, %11 : tensor<1x16x128xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %15 = stablehlo.multiply %14, %12 : tensor<1x16x128xf32>
    %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %17 = stablehlo.add %10, %16 : tensor<1x16x1xf32>
    %18 = stablehlo.sqrt %17 : tensor<1x16x1xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %20 = stablehlo.divide %15, %19 : tensor<1x16x128xf32>
    %21 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %23 = stablehlo.add %20, %22 : tensor<1x16x128xf32>
    return %23 : tensor<1x16x128xf32>
  }

// ── Causal Self-Attention ──
  func.func @causal_self_attention(%arg0: tensor<1x16x128xf32>, %arg1: tensor<128x384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<128x128xf32>, %arg4: tensor<128xf32>) -> (tensor<1x16x128xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x384xf32>) -> tensor<1x16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x16x384xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x384xf32>
    %4 = stablehlo.slice %3 [0:1, 0:16, 0:128] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %5 = stablehlo.slice %3 [0:1, 0:16, 128:256] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %6 = stablehlo.slice %3 [0:1, 0:16, 256:384] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %7 = stablehlo.reshape %4 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %8 = stablehlo.transpose %7, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %9 = stablehlo.reshape %5 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %11 = stablehlo.reshape %6 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %cst = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %13 = stablehlo.sqrt %cst : tensor<f32>
    %14 = stablehlo.transpose %10, dims = [0, 1, 3, 2] : (tensor<1x4x16x32xf32>) -> tensor<1x4x32x16xf32>
    %15 = stablehlo.reshape %8 : (tensor<1x4x16x32xf32>) -> tensor<4x16x32xf32>
    %16 = stablehlo.dot_general %15, %14, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x32xf32>, tensor<1x4x32x16xf32>) -> tensor<4x16x1x16xf32>
    %17 = stablehlo.transpose %16, dims = [2, 0, 1, 3] : (tensor<4x16x1x16xf32>) -> tensor<1x4x16x16xf32>
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<1x4x16x16xf32>
    %19 = stablehlo.divide %17, %18 : tensor<1x4x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %21 = call @causal_self_attention__tril(%20) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %23 = stablehlo.compare  EQ, %21, %22,  FLOAT : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
    %cst_2 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %24 = call @causal_self_attention___where(%23, %cst_2, %19) : (tensor<16x16xi1>, tensor<f32>, tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %25 = stablehlo.reduce(%24 init: %cst_3) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %cst_4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x16xf32>
    %27 = stablehlo.maximum %26, %25 : tensor<1x4x16xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %30 = stablehlo.subtract %24, %29 : tensor<1x4x16x16xf32>
    %31 = stablehlo.exponential %30 : tensor<1x4x16x16xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%31 init: %cst_5) applies stablehlo.add across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %35 = stablehlo.divide %31, %34 : tensor<1x4x16x16xf32>
    %36 = stablehlo.reshape %35 : (tensor<1x4x16x16xf32>) -> tensor<4x16x16xf32>
    %37 = stablehlo.dot_general %36, %12, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xf32>, tensor<1x4x16x32xf32>) -> tensor<4x16x1x32xf32>
    %38 = stablehlo.transpose %37, dims = [2, 0, 1, 3] : (tensor<4x16x1x32xf32>) -> tensor<1x4x16x32xf32>
    %39 = stablehlo.transpose %38, dims = [0, 2, 1, 3] : (tensor<1x4x16x32xf32>) -> tensor<1x16x4x32xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x16x4x32xf32>) -> tensor<1x16x128xf32>
    %41 = stablehlo.dot_general %40, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x128xf32>) -> tensor<1x16x128xf32>
    %42 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %44 = stablehlo.add %41, %43 : tensor<1x16x128xf32>
    return %44 : tensor<1x16x128xf32>
  }

  func.func private @causal_self_attention__tril(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    %0 = stablehlo.iota dim = 0 : tensor<16x16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<16x16xi32>
    %3 = stablehlo.iota dim = 1 : tensor<16x16xi32>
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<16x16xi1>, tensor<16x16xf32>
    return %6 : tensor<16x16xf32>
  }

  func.func private @causal_self_attention___where(%arg0: tensor<16x16xi1>, %arg1: tensor<f32>, %arg2: tensor<1x4x16x16xf32>) -> (tensor<1x4x16x16xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 3] : (tensor<16x16xi1>) -> tensor<1x4x16x16xi1>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x4x16x16xf32>
    %2 = stablehlo.select %0, %1, %arg2 : tensor<1x4x16x16xi1>, tensor<1x4x16x16xf32>
    return %2 : tensor<1x4x16x16xf32>
  }

// ── Feed-Forward MLP ──
  func.func @mlp_block(%arg0: tensor<1x16x128xf32>, %arg1: tensor<128x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x128xf32>, %arg4: tensor<128xf32>) -> (tensor<1x16x128xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x512xf32>) -> tensor<1x16x512xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x16x512xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x512xf32>
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x16x512xf32>
    %cst_0 = stablehlo.constant dense<0.636619746> : tensor<f32>
    %6 = stablehlo.sqrt %cst_0 : tensor<f32>
    %7 = stablehlo.multiply %3, %3 : tensor<1x16x512xf32>
    %8 = stablehlo.multiply %7, %3 : tensor<1x16x512xf32>
    %cst_1 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %10 = stablehlo.multiply %9, %8 : tensor<1x16x512xf32>
    %11 = stablehlo.add %3, %10 : tensor<1x16x512xf32>
    %12 = stablehlo.convert %6 : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %14 = stablehlo.multiply %13, %11 : tensor<1x16x512xf32>
    %15 = stablehlo.tanh %14 : tensor<1x16x512xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %17 = stablehlo.add %16, %15 : tensor<1x16x512xf32>
    %18 = stablehlo.multiply %5, %17 : tensor<1x16x512xf32>
    %19 = stablehlo.dot_general %18, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x512xf32>, tensor<512x128xf32>) -> tensor<1x16x128xf32>
    %20 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %22 = stablehlo.add %19, %21 : tensor<1x16x128xf32>
    return %22 : tensor<1x16x128xf32>
  }

// ── Transformer Block (attention + mlp with residuals) ──
  func.func @transformer_block(%arg0: tensor<1x16x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x384xf32>, %arg4: tensor<384xf32>, %arg5: tensor<128x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x512xf32>, %arg10: tensor<512xf32>, %arg11: tensor<512x128xf32>, %arg12: tensor<128xf32>) -> (tensor<1x16x128xf32>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_0 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %3 = stablehlo.divide %1, %2 : tensor<1x16x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x16x128xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x16x128xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_2 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %10 = stablehlo.divide %8, %9 : tensor<1x16x1xf32>
    %11 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %12 = stablehlo.subtract %arg0, %11 : tensor<1x16x128xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %15 = stablehlo.multiply %14, %12 : tensor<1x16x128xf32>
    %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %17 = stablehlo.add %10, %16 : tensor<1x16x1xf32>
    %18 = stablehlo.sqrt %17 : tensor<1x16x1xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %20 = stablehlo.divide %15, %19 : tensor<1x16x128xf32>
    %21 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %23 = stablehlo.add %20, %22 : tensor<1x16x128xf32>
    %24 = stablehlo.dot_general %23, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x384xf32>) -> tensor<1x16x384xf32>
    %25 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x16x384xf32>
    %27 = stablehlo.add %24, %26 : tensor<1x16x384xf32>
    %28 = stablehlo.slice %27 [0:1, 0:16, 0:128] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %29 = stablehlo.slice %27 [0:1, 0:16, 128:256] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %30 = stablehlo.slice %27 [0:1, 0:16, 256:384] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %31 = stablehlo.reshape %28 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %32 = stablehlo.transpose %31, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %33 = stablehlo.reshape %29 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %34 = stablehlo.transpose %33, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %35 = stablehlo.reshape %30 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %36 = stablehlo.transpose %35, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %cst_4 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %37 = stablehlo.sqrt %cst_4 : tensor<f32>
    %38 = stablehlo.transpose %34, dims = [0, 1, 3, 2] : (tensor<1x4x16x32xf32>) -> tensor<1x4x32x16xf32>
    %39 = stablehlo.reshape %32 : (tensor<1x4x16x32xf32>) -> tensor<4x16x32xf32>
    %40 = stablehlo.dot_general %39, %38, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x32xf32>, tensor<1x4x32x16xf32>) -> tensor<4x16x1x16xf32>
    %41 = stablehlo.transpose %40, dims = [2, 0, 1, 3] : (tensor<4x16x1x16xf32>) -> tensor<1x4x16x16xf32>
    %42 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<1x4x16x16xf32>
    %43 = stablehlo.divide %41, %42 : tensor<1x4x16x16xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %45 = call @transformer_block__tril(%44) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %47 = stablehlo.compare  EQ, %45, %46,  FLOAT : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
    %cst_7 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %48 = call @transformer_block___where(%47, %cst_7, %43) : (tensor<16x16xi1>, tensor<f32>, tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32>
    %cst_8 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %49 = stablehlo.reduce(%48 init: %cst_8) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %cst_9 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x4x16xf32>
    %51 = stablehlo.maximum %50, %49 : tensor<1x4x16xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %54 = stablehlo.subtract %48, %53 : tensor<1x4x16x16xf32>
    %55 = stablehlo.exponential %54 : tensor<1x4x16x16xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %56 = stablehlo.reduce(%55 init: %cst_10) applies stablehlo.add across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %59 = stablehlo.divide %55, %58 : tensor<1x4x16x16xf32>
    %60 = stablehlo.reshape %59 : (tensor<1x4x16x16xf32>) -> tensor<4x16x16xf32>
    %61 = stablehlo.dot_general %60, %36, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xf32>, tensor<1x4x16x32xf32>) -> tensor<4x16x1x32xf32>
    %62 = stablehlo.transpose %61, dims = [2, 0, 1, 3] : (tensor<4x16x1x32xf32>) -> tensor<1x4x16x32xf32>
    %63 = stablehlo.transpose %62, dims = [0, 2, 1, 3] : (tensor<1x4x16x32xf32>) -> tensor<1x16x4x32xf32>
    %64 = stablehlo.reshape %63 : (tensor<1x16x4x32xf32>) -> tensor<1x16x128xf32>
    %65 = stablehlo.dot_general %64, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x128xf32>) -> tensor<1x16x128xf32>
    %66 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %68 = stablehlo.add %65, %67 : tensor<1x16x128xf32>
    %69 = stablehlo.add %arg0, %68 : tensor<1x16x128xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %70 = stablehlo.reduce(%69 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_12 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %73 = stablehlo.divide %71, %72 : tensor<1x16x1xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %75 = stablehlo.subtract %69, %74 : tensor<1x16x128xf32>
    %76 = stablehlo.multiply %75, %75 : tensor<1x16x128xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %77 = stablehlo.reduce(%76 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_14 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %79 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %80 = stablehlo.divide %78, %79 : tensor<1x16x1xf32>
    %81 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %82 = stablehlo.subtract %69, %81 : tensor<1x16x128xf32>
    %83 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %85 = stablehlo.multiply %84, %82 : tensor<1x16x128xf32>
    %cst_15 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %87 = stablehlo.add %80, %86 : tensor<1x16x1xf32>
    %88 = stablehlo.sqrt %87 : tensor<1x16x1xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %90 = stablehlo.divide %85, %89 : tensor<1x16x128xf32>
    %91 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %93 = stablehlo.add %90, %92 : tensor<1x16x128xf32>
    %94 = stablehlo.dot_general %93, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x512xf32>) -> tensor<1x16x512xf32>
    %95 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x16x512xf32>
    %97 = stablehlo.add %94, %96 : tensor<1x16x512xf32>
    %cst_16 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %99 = stablehlo.multiply %97, %98 : tensor<1x16x512xf32>
    %cst_17 = stablehlo.constant dense<0.636619746> : tensor<f32>
    %100 = stablehlo.sqrt %cst_17 : tensor<f32>
    %101 = stablehlo.multiply %97, %97 : tensor<1x16x512xf32>
    %102 = stablehlo.multiply %101, %97 : tensor<1x16x512xf32>
    %cst_18 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %104 = stablehlo.multiply %103, %102 : tensor<1x16x512xf32>
    %105 = stablehlo.add %97, %104 : tensor<1x16x512xf32>
    %106 = stablehlo.convert %100 : tensor<f32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %108 = stablehlo.multiply %107, %105 : tensor<1x16x512xf32>
    %109 = stablehlo.tanh %108 : tensor<1x16x512xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %110 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %111 = stablehlo.add %110, %109 : tensor<1x16x512xf32>
    %112 = stablehlo.multiply %99, %111 : tensor<1x16x512xf32>
    %113 = stablehlo.dot_general %112, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x512xf32>, tensor<512x128xf32>) -> tensor<1x16x128xf32>
    %114 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %115 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %116 = stablehlo.add %113, %115 : tensor<1x16x128xf32>
    %117 = stablehlo.add %69, %116 : tensor<1x16x128xf32>
    return %117 : tensor<1x16x128xf32>
  }

  func.func private @transformer_block__tril(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    %0 = stablehlo.iota dim = 0 : tensor<16x16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<16x16xi32>
    %3 = stablehlo.iota dim = 1 : tensor<16x16xi32>
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<16x16xi1>, tensor<16x16xf32>
    return %6 : tensor<16x16xf32>
  }

  func.func private @transformer_block___where(%arg0: tensor<16x16xi1>, %arg1: tensor<f32>, %arg2: tensor<1x4x16x16xf32>) -> (tensor<1x4x16x16xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 3] : (tensor<16x16xi1>) -> tensor<1x4x16x16xi1>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x4x16x16xf32>
    %2 = stablehlo.select %0, %1, %arg2 : tensor<1x4x16x16xi1>, tensor<1x4x16x16xf32>
    return %2 : tensor<1x4x16x16xf32>
  }

// ── Language Model Head ──
  func.func @lm_head(%arg0: tensor<1x16x128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>) -> (tensor<1x16x256xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x256xf32>) -> tensor<1x16x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<1x1x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x1x256xf32>) -> tensor<1x16x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x256xf32>
    return %3 : tensor<1x16x256xf32>
  }

// ── Full Forward Pass ──
  func.func @gpt2_forward(%arg0: tensor<256x128xf32>, %arg1: tensor<16x128xf32>, %arg2: tensor<1x16xi32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x384xf32>, %arg6: tensor<384xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x512xf32>, %arg12: tensor<512xf32>, %arg13: tensor<512x128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128x384xf32>, %arg18: tensor<384xf32>, %arg19: tensor<128x128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128x512xf32>, %arg24: tensor<512xf32>, %arg25: tensor<512x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x256xf32>, %arg30: tensor<256xf32>) -> (tensor<1x16x256xf32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %1 = stablehlo.compare  LT, %arg2, %0,  SIGNED : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x16xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<1x16xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<1x16xi1>, tensor<1x16xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x16xi32>) -> tensor<1x16x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<256x128xf32>, tensor<1x16x1xi32>) -> tensor<1x16x128xf32>
    %7 = stablehlo.iota dim = 0 : tensor<16xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %9 = stablehlo.compare  LT, %7, %8,  SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_2 = stablehlo.constant dense<16> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %11 = stablehlo.add %7, %10 : tensor<16xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<16xi1>, tensor<16xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %14 = "stablehlo.gather"(%arg1, %13) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<16x128xf32>, tensor<16x1xi32>) -> tensor<16x128xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<16x128xf32>) -> tensor<1x16x128xf32>
    %16 = stablehlo.add %6, %15 : tensor<1x16x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_3 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %20 = stablehlo.divide %18, %19 : tensor<1x16x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %22 = stablehlo.subtract %16, %21 : tensor<1x16x128xf32>
    %23 = stablehlo.multiply %22, %22 : tensor<1x16x128xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst_4) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_5 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %27 = stablehlo.divide %25, %26 : tensor<1x16x1xf32>
    %28 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %29 = stablehlo.subtract %16, %28 : tensor<1x16x128xf32>
    %30 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %32 = stablehlo.multiply %31, %29 : tensor<1x16x128xf32>
    %cst_6 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %34 = stablehlo.add %27, %33 : tensor<1x16x1xf32>
    %35 = stablehlo.sqrt %34 : tensor<1x16x1xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %37 = stablehlo.divide %32, %36 : tensor<1x16x128xf32>
    %38 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %40 = stablehlo.add %37, %39 : tensor<1x16x128xf32>
    %41 = stablehlo.dot_general %40, %arg5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x384xf32>) -> tensor<1x16x384xf32>
    %42 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x16x384xf32>
    %44 = stablehlo.add %41, %43 : tensor<1x16x384xf32>
    %45 = stablehlo.slice %44 [0:1, 0:16, 0:128] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %46 = stablehlo.slice %44 [0:1, 0:16, 128:256] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %47 = stablehlo.slice %44 [0:1, 0:16, 256:384] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %48 = stablehlo.reshape %45 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %49 = stablehlo.transpose %48, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %50 = stablehlo.reshape %46 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %51 = stablehlo.transpose %50, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %52 = stablehlo.reshape %47 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %53 = stablehlo.transpose %52, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %cst_7 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %54 = stablehlo.sqrt %cst_7 : tensor<f32>
    %55 = stablehlo.transpose %51, dims = [0, 1, 3, 2] : (tensor<1x4x16x32xf32>) -> tensor<1x4x32x16xf32>
    %56 = stablehlo.reshape %49 : (tensor<1x4x16x32xf32>) -> tensor<4x16x32xf32>
    %57 = stablehlo.dot_general %56, %55, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x32xf32>, tensor<1x4x32x16xf32>) -> tensor<4x16x1x16xf32>
    %58 = stablehlo.transpose %57, dims = [2, 0, 1, 3] : (tensor<4x16x1x16xf32>) -> tensor<1x4x16x16xf32>
    %59 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<f32>) -> tensor<1x4x16x16xf32>
    %60 = stablehlo.divide %58, %59 : tensor<1x4x16x16xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %62 = call @gpt2_forward__tril(%61) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %64 = stablehlo.compare  EQ, %62, %63,  FLOAT : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
    %cst_10 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %65 = call @gpt2_forward___where(%64, %cst_10, %60) : (tensor<16x16xi1>, tensor<f32>, tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32>
    %cst_11 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %66 = stablehlo.reduce(%65 init: %cst_11) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %cst_12 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x4x16xf32>
    %68 = stablehlo.maximum %67, %66 : tensor<1x4x16xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %71 = stablehlo.subtract %65, %70 : tensor<1x4x16x16xf32>
    %72 = stablehlo.exponential %71 : tensor<1x4x16x16xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %73 = stablehlo.reduce(%72 init: %cst_13) applies stablehlo.add across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %76 = stablehlo.divide %72, %75 : tensor<1x4x16x16xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x4x16x16xf32>) -> tensor<4x16x16xf32>
    %78 = stablehlo.dot_general %77, %53, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xf32>, tensor<1x4x16x32xf32>) -> tensor<4x16x1x32xf32>
    %79 = stablehlo.transpose %78, dims = [2, 0, 1, 3] : (tensor<4x16x1x32xf32>) -> tensor<1x4x16x32xf32>
    %80 = stablehlo.transpose %79, dims = [0, 2, 1, 3] : (tensor<1x4x16x32xf32>) -> tensor<1x16x4x32xf32>
    %81 = stablehlo.reshape %80 : (tensor<1x16x4x32xf32>) -> tensor<1x16x128xf32>
    %82 = stablehlo.dot_general %81, %arg7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x128xf32>) -> tensor<1x16x128xf32>
    %83 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %85 = stablehlo.add %82, %84 : tensor<1x16x128xf32>
    %86 = stablehlo.add %16, %85 : tensor<1x16x128xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %87 = stablehlo.reduce(%86 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_15 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %90 = stablehlo.divide %88, %89 : tensor<1x16x1xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %92 = stablehlo.subtract %86, %91 : tensor<1x16x128xf32>
    %93 = stablehlo.multiply %92, %92 : tensor<1x16x128xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %94 = stablehlo.reduce(%93 init: %cst_16) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_17 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %96 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %97 = stablehlo.divide %95, %96 : tensor<1x16x1xf32>
    %98 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %99 = stablehlo.subtract %86, %98 : tensor<1x16x128xf32>
    %100 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %102 = stablehlo.multiply %101, %99 : tensor<1x16x128xf32>
    %cst_18 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %104 = stablehlo.add %97, %103 : tensor<1x16x1xf32>
    %105 = stablehlo.sqrt %104 : tensor<1x16x1xf32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %107 = stablehlo.divide %102, %106 : tensor<1x16x128xf32>
    %108 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %110 = stablehlo.add %107, %109 : tensor<1x16x128xf32>
    %111 = stablehlo.dot_general %110, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x512xf32>) -> tensor<1x16x512xf32>
    %112 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x16x512xf32>
    %114 = stablehlo.add %111, %113 : tensor<1x16x512xf32>
    %cst_19 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %115 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %116 = stablehlo.multiply %114, %115 : tensor<1x16x512xf32>
    %cst_20 = stablehlo.constant dense<0.636619746> : tensor<f32>
    %117 = stablehlo.sqrt %cst_20 : tensor<f32>
    %118 = stablehlo.multiply %114, %114 : tensor<1x16x512xf32>
    %119 = stablehlo.multiply %118, %114 : tensor<1x16x512xf32>
    %cst_21 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %120 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %121 = stablehlo.multiply %120, %119 : tensor<1x16x512xf32>
    %122 = stablehlo.add %114, %121 : tensor<1x16x512xf32>
    %123 = stablehlo.convert %117 : tensor<f32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %125 = stablehlo.multiply %124, %122 : tensor<1x16x512xf32>
    %126 = stablehlo.tanh %125 : tensor<1x16x512xf32>
    %cst_22 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %127 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %128 = stablehlo.add %127, %126 : tensor<1x16x512xf32>
    %129 = stablehlo.multiply %116, %128 : tensor<1x16x512xf32>
    %130 = stablehlo.dot_general %129, %arg13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x512xf32>, tensor<512x128xf32>) -> tensor<1x16x128xf32>
    %131 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %133 = stablehlo.add %130, %132 : tensor<1x16x128xf32>
    %134 = stablehlo.add %86, %133 : tensor<1x16x128xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %135 = stablehlo.reduce(%134 init: %cst_23) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_24 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %137 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %138 = stablehlo.divide %136, %137 : tensor<1x16x1xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %140 = stablehlo.subtract %134, %139 : tensor<1x16x128xf32>
    %141 = stablehlo.multiply %140, %140 : tensor<1x16x128xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %142 = stablehlo.reduce(%141 init: %cst_25) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_26 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %144 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %145 = stablehlo.divide %143, %144 : tensor<1x16x1xf32>
    %146 = stablehlo.broadcast_in_dim %138, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %147 = stablehlo.subtract %134, %146 : tensor<1x16x128xf32>
    %148 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %150 = stablehlo.multiply %149, %147 : tensor<1x16x128xf32>
    %cst_27 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %151 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %152 = stablehlo.add %145, %151 : tensor<1x16x1xf32>
    %153 = stablehlo.sqrt %152 : tensor<1x16x1xf32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %155 = stablehlo.divide %150, %154 : tensor<1x16x128xf32>
    %156 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %158 = stablehlo.add %155, %157 : tensor<1x16x128xf32>
    %159 = stablehlo.dot_general %158, %arg17, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x384xf32>) -> tensor<1x16x384xf32>
    %160 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x16x384xf32>
    %162 = stablehlo.add %159, %161 : tensor<1x16x384xf32>
    %163 = stablehlo.slice %162 [0:1, 0:16, 0:128] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %164 = stablehlo.slice %162 [0:1, 0:16, 128:256] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %165 = stablehlo.slice %162 [0:1, 0:16, 256:384] : (tensor<1x16x384xf32>) -> tensor<1x16x128xf32>
    %166 = stablehlo.reshape %163 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %167 = stablehlo.transpose %166, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %168 = stablehlo.reshape %164 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %169 = stablehlo.transpose %168, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %170 = stablehlo.reshape %165 : (tensor<1x16x128xf32>) -> tensor<1x16x4x32xf32>
    %171 = stablehlo.transpose %170, dims = [0, 2, 1, 3] : (tensor<1x16x4x32xf32>) -> tensor<1x4x16x32xf32>
    %cst_28 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %172 = stablehlo.sqrt %cst_28 : tensor<f32>
    %173 = stablehlo.transpose %169, dims = [0, 1, 3, 2] : (tensor<1x4x16x32xf32>) -> tensor<1x4x32x16xf32>
    %174 = stablehlo.reshape %167 : (tensor<1x4x16x32xf32>) -> tensor<4x16x32xf32>
    %175 = stablehlo.dot_general %174, %173, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x32xf32>, tensor<1x4x32x16xf32>) -> tensor<4x16x1x16xf32>
    %176 = stablehlo.transpose %175, dims = [2, 0, 1, 3] : (tensor<4x16x1x16xf32>) -> tensor<1x4x16x16xf32>
    %177 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f32>) -> tensor<1x4x16x16xf32>
    %178 = stablehlo.divide %176, %177 : tensor<1x4x16x16xf32>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %179 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %180 = call @gpt2_forward__tril(%179) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %181 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %182 = stablehlo.compare  EQ, %180, %181,  FLOAT : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xi1>
    %cst_31 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %183 = call @gpt2_forward___where(%182, %cst_31, %178) : (tensor<16x16xi1>, tensor<f32>, tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32>
    %cst_32 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %184 = stablehlo.reduce(%183 init: %cst_32) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %cst_33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %185 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x4x16xf32>
    %186 = stablehlo.maximum %185, %184 : tensor<1x4x16xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %189 = stablehlo.subtract %183, %188 : tensor<1x4x16x16xf32>
    %190 = stablehlo.exponential %189 : tensor<1x4x16x16xf32>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %191 = stablehlo.reduce(%190 init: %cst_34) applies stablehlo.add across dimensions = [3] : (tensor<1x4x16x16xf32>, tensor<f32>) -> tensor<1x4x16xf32>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2] : (tensor<1x4x16xf32>) -> tensor<1x4x16x1xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x4x16x1xf32>) -> tensor<1x4x16x16xf32>
    %194 = stablehlo.divide %190, %193 : tensor<1x4x16x16xf32>
    %195 = stablehlo.reshape %194 : (tensor<1x4x16x16xf32>) -> tensor<4x16x16xf32>
    %196 = stablehlo.dot_general %195, %171, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xf32>, tensor<1x4x16x32xf32>) -> tensor<4x16x1x32xf32>
    %197 = stablehlo.transpose %196, dims = [2, 0, 1, 3] : (tensor<4x16x1x32xf32>) -> tensor<1x4x16x32xf32>
    %198 = stablehlo.transpose %197, dims = [0, 2, 1, 3] : (tensor<1x4x16x32xf32>) -> tensor<1x16x4x32xf32>
    %199 = stablehlo.reshape %198 : (tensor<1x16x4x32xf32>) -> tensor<1x16x128xf32>
    %200 = stablehlo.dot_general %199, %arg19, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x128xf32>) -> tensor<1x16x128xf32>
    %201 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %203 = stablehlo.add %200, %202 : tensor<1x16x128xf32>
    %204 = stablehlo.add %134, %203 : tensor<1x16x128xf32>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %205 = stablehlo.reduce(%204 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_36 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %207 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %208 = stablehlo.divide %206, %207 : tensor<1x16x1xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %210 = stablehlo.subtract %204, %209 : tensor<1x16x128xf32>
    %211 = stablehlo.multiply %210, %210 : tensor<1x16x128xf32>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %212 = stablehlo.reduce(%211 init: %cst_37) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_38 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %214 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %215 = stablehlo.divide %213, %214 : tensor<1x16x1xf32>
    %216 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %217 = stablehlo.subtract %204, %216 : tensor<1x16x128xf32>
    %218 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %220 = stablehlo.multiply %219, %217 : tensor<1x16x128xf32>
    %cst_39 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %221 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %222 = stablehlo.add %215, %221 : tensor<1x16x1xf32>
    %223 = stablehlo.sqrt %222 : tensor<1x16x1xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %225 = stablehlo.divide %220, %224 : tensor<1x16x128xf32>
    %226 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %228 = stablehlo.add %225, %227 : tensor<1x16x128xf32>
    %229 = stablehlo.dot_general %228, %arg23, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x512xf32>) -> tensor<1x16x512xf32>
    %230 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x16x512xf32>
    %232 = stablehlo.add %229, %231 : tensor<1x16x512xf32>
    %cst_40 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %233 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %234 = stablehlo.multiply %232, %233 : tensor<1x16x512xf32>
    %cst_41 = stablehlo.constant dense<0.636619746> : tensor<f32>
    %235 = stablehlo.sqrt %cst_41 : tensor<f32>
    %236 = stablehlo.multiply %232, %232 : tensor<1x16x512xf32>
    %237 = stablehlo.multiply %236, %232 : tensor<1x16x512xf32>
    %cst_42 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %238 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %239 = stablehlo.multiply %238, %237 : tensor<1x16x512xf32>
    %240 = stablehlo.add %232, %239 : tensor<1x16x512xf32>
    %241 = stablehlo.convert %235 : tensor<f32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %243 = stablehlo.multiply %242, %240 : tensor<1x16x512xf32>
    %244 = stablehlo.tanh %243 : tensor<1x16x512xf32>
    %cst_43 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %245 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1x16x512xf32>
    %246 = stablehlo.add %245, %244 : tensor<1x16x512xf32>
    %247 = stablehlo.multiply %234, %246 : tensor<1x16x512xf32>
    %248 = stablehlo.dot_general %247, %arg25, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x512xf32>, tensor<512x128xf32>) -> tensor<1x16x128xf32>
    %249 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %251 = stablehlo.add %248, %250 : tensor<1x16x128xf32>
    %252 = stablehlo.add %204, %251 : tensor<1x16x128xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %253 = stablehlo.reduce(%252 init: %cst_44) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %254 = stablehlo.broadcast_in_dim %253, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_45 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %255 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %256 = stablehlo.divide %254, %255 : tensor<1x16x1xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %258 = stablehlo.subtract %252, %257 : tensor<1x16x128xf32>
    %259 = stablehlo.multiply %258, %258 : tensor<1x16x128xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %260 = stablehlo.reduce(%259 init: %cst_46) applies stablehlo.add across dimensions = [2] : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16xf32>
    %261 = stablehlo.broadcast_in_dim %260, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
    %cst_47 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %262 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %263 = stablehlo.divide %261, %262 : tensor<1x16x1xf32>
    %264 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %265 = stablehlo.subtract %252, %264 : tensor<1x16x128xf32>
    %266 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %268 = stablehlo.multiply %267, %265 : tensor<1x16x128xf32>
    %cst_48 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %269 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
    %270 = stablehlo.add %263, %269 : tensor<1x16x1xf32>
    %271 = stablehlo.sqrt %270 : tensor<1x16x1xf32>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x128xf32>
    %273 = stablehlo.divide %268, %272 : tensor<1x16x128xf32>
    %274 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2] : (tensor<1x1x128xf32>) -> tensor<1x16x128xf32>
    %276 = stablehlo.add %273, %275 : tensor<1x16x128xf32>
    %277 = stablehlo.dot_general %276, %arg29, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16x128xf32>, tensor<128x256xf32>) -> tensor<1x16x256xf32>
    %278 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<256xf32>) -> tensor<1x1x256xf32>
    %279 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2] : (tensor<1x1x256xf32>) -> tensor<1x16x256xf32>
    %280 = stablehlo.add %277, %279 : tensor<1x16x256xf32>
    return %280 : tensor<1x16x256xf32>
  }

  func.func private @gpt2_forward__tril(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    %0 = stablehlo.iota dim = 0 : tensor<16x16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<16x16xi32>
    %3 = stablehlo.iota dim = 1 : tensor<16x16xi32>
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<16x16xi1>, tensor<16x16xf32>
    return %6 : tensor<16x16xf32>
  }

  func.func private @gpt2_forward___where(%arg0: tensor<16x16xi1>, %arg1: tensor<f32>, %arg2: tensor<1x4x16x16xf32>) -> (tensor<1x4x16x16xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 3] : (tensor<16x16xi1>) -> tensor<1x4x16x16xi1>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x4x16x16xf32>
    %2 = stablehlo.select %0, %1, %arg2 : tensor<1x4x16x16xi1>, tensor<1x4x16x16xf32>
    return %2 : tensor<1x4x16x16xf32>
  }

}  // module @gpt2