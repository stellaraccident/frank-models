// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SwiGLU activation: gate * silu(up)
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Usage:
//   %output = call @swiglu(%gate, %up)
//       : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

module @activation_components {

  util.func public @swiglu(
      %gate: tensor<?x?x?xf32>,  // [batch, seq, n_ff]
      %up: tensor<?x?x?xf32>     // [batch, seq, n_ff]
  ) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %gate, %c0 : tensor<?x?x?xf32>
    %d1 = tensor.dim %gate, %c1 : tensor<?x?x?xf32>
    %d2 = tensor.dim %gate, %c2 : tensor<?x?x?xf32>

    %output_init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%gate, %up : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%output_init : tensor<?x?x?xf32>) {
    ^bb0(%g: f32, %u: f32, %out: f32):
      // silu(u) = u * sigmoid(u) = u / (1 + exp(-u))
      %neg_u = arith.negf %u : f32
      %exp_neg = math.exp %neg_u : f32
      %one = arith.constant 1.0 : f32
      %denom = arith.addf %one, %exp_neg : f32
      %sigmoid = arith.divf %one, %denom : f32
      %silu = arith.mulf %u, %sigmoid : f32
      // gate * silu(up)
      %result = arith.mulf %g, %silu : f32
      linalg.yield %result : f32
    } -> tensor<?x?x?xf32>

    util.return %output : tensor<?x?x?xf32>
  }

}
