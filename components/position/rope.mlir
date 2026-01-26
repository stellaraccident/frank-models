// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rotary Position Embeddings (RoPE).
// Applies rotation to adjacent dimension pairs based on position.
//
// Usage:
//   %output = call @rope(%input, %positions, %freq_base, %freq_scale)
//       : (tensor<?x?x?x?xf32>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf32>

module @position_components {

  util.func public @rope(
      %input: tensor<?x?x?x?xf32>,   // [batch, seq_len, n_head, head_dim]
      %positions: tensor<?x?xi64>,    // [batch, seq_len]
      %freq_base: f32,                // typically 10000.0
      %freq_scale: f32                // typically 1.0
  ) -> tensor<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %batch = tensor.dim %input, %c0 : tensor<?x?x?x?xf32>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?x?xf32>
    %n_head = tensor.dim %input, %c2 : tensor<?x?x?x?xf32>
    %head_dim = tensor.dim %input, %c3 : tensor<?x?x?x?xf32>

    // Reshape to expose dimension pairs: [batch, seq_len, n_head, head_dim]
    // -> [batch, seq_len, n_head, head_dim/2, 2]
    %half_dim = arith.divsi %head_dim, %c2 : index
    %reshaped = tensor.expand_shape %input [[0], [1], [2], [3, 4]] output_shape [%batch, %seq_len, %n_head, %half_dim, 2]
        : tensor<?x?x?x?xf32> into tensor<?x?x?x?x2xf32>

    // Compute frequencies for each dimension pair.
    // freq[i] = (1.0 / base^(2i/head_dim)) * freq_scale
    %head_dim_f32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32_cast = arith.sitofp %head_dim_f32 : i32 to f32

    %freq_init = tensor.empty(%half_dim) : tensor<?xf32>
    %freqs = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%freq_init : tensor<?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx_f32 = arith.index_cast %idx : index to i32
      %idx_f32_cast = arith.sitofp %idx_f32 : i32 to f32
      %two = arith.constant 2.0 : f32
      %two_i = arith.mulf %two, %idx_f32_cast : f32
      %exp = arith.divf %two_i, %head_dim_f32_cast : f32
      %neg_exp = arith.negf %exp : f32
      %base_freq = math.powf %freq_base, %neg_exp : f32
      %freq = arith.mulf %base_freq, %freq_scale : f32
      linalg.yield %freq : f32
    } -> tensor<?xf32>

    // Apply RoPE rotation to paired elements.
    %output_init = tensor.empty(%batch, %seq_len, %n_head, %half_dim) : tensor<?x?x?x?x2xf32>
    %rotated = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, 0)>,  // x0 (first of pair)
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, 1)>,  // x1 (second of pair)
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,             // position
        affine_map<(d0, d1, d2, d3, d4) -> (d3)>,                 // frequency
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>  // output
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%reshaped, %reshaped, %positions, %freqs : tensor<?x?x?x?x2xf32>, tensor<?x?x?x?x2xf32>, tensor<?x?xi64>, tensor<?xf32>)
      outs(%output_init : tensor<?x?x?x?x2xf32>) {
    ^bb0(%x0: f32, %x1: f32, %pos_i64: i64, %freq: f32, %out: f32):
      // Compute angle = position * frequency.
      %pos_i32 = arith.trunci %pos_i64 : i64 to i32
      %pos_f32 = arith.sitofp %pos_i32 : i32 to f32
      %angle = arith.mulf %pos_f32, %freq : f32

      %cos_val = math.cos %angle : f32
      %sin_val = math.sin %angle : f32

      // Apply rotation matrix to pair.
      // x0' = x0*cos - x1*sin
      // x1' = x0*sin + x1*cos
      %pair_idx = linalg.index 4 : index
      %c0_idx = arith.constant 0 : index
      %is_first = arith.cmpi eq, %pair_idx, %c0_idx : index

      %result = scf.if %is_first -> (f32) {
        // First element: x0' = x0*cos - x1*sin
        %x0_cos = arith.mulf %x0, %cos_val : f32
        %x1_sin = arith.mulf %x1, %sin_val : f32
        %rotated = arith.subf %x0_cos, %x1_sin : f32
        scf.yield %rotated : f32
      } else {
        // Second element: x1' = x0*sin + x1*cos
        %x0_sin = arith.mulf %x0, %sin_val : f32
        %x1_cos = arith.mulf %x1, %cos_val : f32
        %rotated = arith.addf %x0_sin, %x1_cos : f32
        scf.yield %rotated : f32
      }

      linalg.yield %result : f32
    } -> tensor<?x?x?x?x2xf32>

    // Collapse back to original shape.
    %output = tensor.collapse_shape %rotated [[0], [1], [2], [3, 4]] : tensor<?x?x?x?x2xf32> into tensor<?x?x?x?xf32>

    util.return %output : tensor<?x?x?x?xf32>
  }

}
