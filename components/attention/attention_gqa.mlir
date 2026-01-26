// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Multi-Head Attention using iree_linalg_ext.attention.
// Note: Currently requires n_head == n_head_kv (MHA, not GQA).
// GQA support requires explicit KV head repetition (future work).
//
// Usage:
//   %output = call @attention_gqa(%query, %key, %value, %scale)
//       : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32)
//       -> tensor<?x?x?x?xf32>

module @attention_components {

  util.func public @attention_gqa(
      %query: tensor<?x?x?x?xf32>,   // [batch, seq_len, n_head, head_dim]
      %key: tensor<?x?x?x?xf32>,     // [batch, seq_len, n_head, head_dim]
      %value: tensor<?x?x?x?xf32>,   // [batch, seq_len, n_head, head_dim]
      %scale: f32                     // 1.0 / sqrt(head_dim)
  ) -> tensor<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %batch = tensor.dim %query, %c0 : tensor<?x?x?x?xf32>
    %seq_len = tensor.dim %query, %c1 : tensor<?x?x?x?xf32>
    %n_head = tensor.dim %query, %c2 : tensor<?x?x?x?xf32>
    %head_dim = tensor.dim %query, %c3 : tensor<?x?x?x?xf32>

    // Transpose Q/K/V from [batch, seq, n_head, head_dim] to [batch, n_head, seq, head_dim]
    %q_transposed_init = tensor.empty(%batch, %n_head, %seq_len, %head_dim) : tensor<?x?x?x?xf32>
    %q_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%query : tensor<?x?x?x?xf32>) outs(%q_transposed_init : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?xf32>

    %k_transposed_init = tensor.empty(%batch, %n_head, %seq_len, %head_dim) : tensor<?x?x?x?xf32>
    %k_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%key : tensor<?x?x?x?xf32>) outs(%k_transposed_init : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?xf32>

    %v_transposed_init = tensor.empty(%batch, %n_head, %seq_len, %head_dim) : tensor<?x?x?x?xf32>
    %v_transposed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%value : tensor<?x?x?x?xf32>) outs(%v_transposed_init : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?xf32>

    // Collapse [batch, n_head, seq, head_dim] to [batch * n_head, seq, head_dim]
    %batch_heads = arith.muli %batch, %n_head : index
    %q_3d = tensor.collapse_shape %q_transposed [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
    %k_3d = tensor.collapse_shape %k_transposed [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
    %v_3d = tensor.collapse_shape %v_transposed [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>

    // Run attention: [batch * n_head, seq, head_dim]
    %output_3d_init = tensor.empty(%batch_heads, %seq_len, %head_dim) : tensor<?x?x?xf32>
    %output_3d = iree_linalg_ext.attention {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,  // Query
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,  // Key
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,  // Value
        affine_map<(d0, d1, d2, d3, d4) -> ()>,            // Scale
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>   // Output
      ]
    } ins(%q_3d, %k_3d, %v_3d, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32)
      outs(%output_3d_init : tensor<?x?x?xf32>) {
    ^bb0(%arg0: f32):
      iree_linalg_ext.yield %arg0 : f32
    } -> tensor<?x?x?xf32>

    // Expand back to [batch, n_head, seq, head_dim]
    %output_4d = tensor.expand_shape %output_3d [[0, 1], [2], [3]]
        output_shape [%batch, %n_head, %seq_len, %head_dim]
        : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>

    // Transpose back from [batch, n_head, seq, head_dim] to [batch, seq, n_head, head_dim]
    %output_init = tensor.empty(%batch, %seq_len, %n_head, %head_dim) : tensor<?x?x?x?xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%output_4d : tensor<?x?x?x?xf32>) outs(%output_init : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?x?xf32>

    util.return %output : tensor<?x?x?x?xf32>
  }

}
