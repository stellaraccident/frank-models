// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Decode Attention Block: QKV projection, RoPE, attention with cached K/V, output projection.
// Processes a single token per sequence using cached K/V from previous tokens.
//
// This is the decode variant that processes one new token and uses cached K/V
// from the KV cache. Used during autoregressive generation.
//
// Logical shapes:
//   input:      [batch, n_embd]               - single token hidden state per sequence
//   positions:  [batch]                       - single position per sequence
//   k_cached:   [batch, ctx_len, n_head_kv, head_dim] - gathered past K (with RoPE)
//   v_cached:   [batch, ctx_len, n_head_kv, head_dim] - gathered past V
//   wq/wk/wv/wo: same as prefill
//
// Returns:
//   output:    [batch, n_embd]                - attention output for single token
//   k_new:     [batch, n_head_kv, head_dim]   - new K with RoPE (for scatter to cache)
//   v_new:     [batch, n_head_kv, head_dim]   - new V (for scatter to cache)

module @attention_block_decode_components {

  // External dependencies resolved by iree-link.
  util.func private @position_components.rope(
      tensor<?x?x?x?xf32>,   // [batch, seq_len, n_head, head_dim]
      tensor<?x?xi64>,        // [batch, seq_len]
      f32,                    // freq_base
      f32                     // freq_scale
  ) -> tensor<?x?x?x?xf32>

  util.func private @attention_components.attention_gqa(
      tensor<?x?x?x?xf32>,   // Q: [batch, seq_len, n_head, head_dim]
      tensor<?x?x?x?xf32>,   // K: [batch, seq_len, n_head_kv, head_dim]
      tensor<?x?x?x?xf32>,   // V: [batch, seq_len, n_head_kv, head_dim]
      f32                     // scale
  ) -> tensor<?x?x?x?xf32>

  util.func public @attention_block_decode(
      %input: tensor<?x?xf32>,              // [batch, n_embd]
      %positions: tensor<?xi64>,             // [batch] - single position per sequence
      %k_cached: tensor<?x?x?x?xf32>,       // [batch, ctx_len, n_head_kv, head_dim]
      %v_cached: tensor<?x?x?x?xf32>,       // [batch, ctx_len, n_head_kv, head_dim]
      %wq: tensor<?x?xf32>,                  // [n_embd, n_embd]
      %wk: tensor<?x?xf32>,                  // [n_embd, n_embd_kv]
      %wv: tensor<?x?xf32>,                  // [n_embd, n_embd_kv]
      %wo: tensor<?x?xf32>,                  // [n_embd, n_embd]
      %bq: tensor<?xf32>,                    // [n_embd] - may be dummy if not used
      %bk: tensor<?xf32>,                    // [n_embd_kv]
      %bv: tensor<?xf32>,                    // [n_embd_kv]
      %bo: tensor<?xf32>,                    // [n_embd]
      %use_bias: i1,                         // flag to enable/disable biases
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %rope_freq_base: f32,
      %rope_freq_scale: f32
  ) -> (tensor<?x?xf32>,                     // output: [batch, n_embd]
        tensor<?x?x?xf32>,                   // k_new: [batch, n_head_kv, head_dim]
        tensor<?x?x?xf32>) {                 // v_new: [batch, n_head_kv, head_dim]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?xf32>

    // Get context length from cached K/V.
    %ctx_len = tensor.dim %k_cached, %c1 : tensor<?x?x?x?xf32>

    // Get dimensions from weights.
    %n_embd_kv = tensor.dim %wk, %c1 : tensor<?x?xf32>

    // Compute head dimensions.
    %head_dim = arith.divsi %n_embd, %n_head : index
    %head_dim_kv = arith.divsi %n_embd_kv, %n_head_kv : index

    // seq_len=1 for decode (single token)
    %seq_len_1 = arith.constant 1 : index

    // Expand input from [batch, n_embd] to [batch, 1, n_embd] for matmul.
    %input_3d = tensor.expand_shape %input [[0], [1, 2]] output_shape [%batch, %seq_len_1, %n_embd]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    // Expand positions from [batch] to [batch, 1] for RoPE.
    %positions_2d = tensor.expand_shape %positions [[0, 1]] output_shape [%batch, %seq_len_1]
        : tensor<?xi64> into tensor<?x?xi64>

    // QKV projections: [batch, 1, n_embd] @ [n_embd, n_out] -> [batch, 1, n_out]
    %cst_zero = arith.constant 0.0 : f32

    // Q projection.
    %q_proj_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf32>
    %q_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%q_proj_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %q_proj = linalg.matmul ins(%input, %wq : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%q_proj_zero : tensor<?x?xf32>) -> tensor<?x?xf32>

    // K projection.
    %k_proj_init = tensor.empty(%batch, %n_embd_kv) : tensor<?x?xf32>
    %k_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%k_proj_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %k_proj = linalg.matmul ins(%input, %wk : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%k_proj_zero : tensor<?x?xf32>) -> tensor<?x?xf32>

    // V projection.
    %v_proj_init = tensor.empty(%batch, %n_embd_kv) : tensor<?x?xf32>
    %v_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%v_proj_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %v_proj = linalg.matmul ins(%input, %wv : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%v_proj_zero : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Conditionally add biases if enabled.
    // bias_add: out[b, i] = proj[b, i] + bias[i]
    %q_final = scf.if %use_bias -> (tensor<?x?xf32>) {
      %q_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,  // proj: [batch, n_embd]
          affine_map<(d0, d1) -> (d1)>,       // bias: [n_embd]
          affine_map<(d0, d1) -> (d0, d1)>   // out: [batch, n_embd]
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%q_proj, %bq : tensor<?x?xf32>, tensor<?xf32>) outs(%q_proj_init : tensor<?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?xf32>
      scf.yield %q_biased : tensor<?x?xf32>
    } else {
      scf.yield %q_proj : tensor<?x?xf32>
    }

    %k_final = scf.if %use_bias -> (tensor<?x?xf32>) {
      %k_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%k_proj, %bk : tensor<?x?xf32>, tensor<?xf32>) outs(%k_proj_init : tensor<?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?xf32>
      scf.yield %k_biased : tensor<?x?xf32>
    } else {
      scf.yield %k_proj : tensor<?x?xf32>
    }

    %v_final = scf.if %use_bias -> (tensor<?x?xf32>) {
      %v_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%v_proj, %bv : tensor<?x?xf32>, tensor<?xf32>) outs(%v_proj_init : tensor<?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?xf32>
      scf.yield %v_biased : tensor<?x?xf32>
    } else {
      scf.yield %v_proj : tensor<?x?xf32>
    }

    // Reshape for multi-head: [batch, n_embd] -> [batch, n_head, head_dim]
    %q_reshaped_3d = tensor.expand_shape %q_final [[0], [1, 2]] output_shape [%batch, %n_head, %head_dim]
        : tensor<?x?xf32> into tensor<?x?x?xf32>
    %k_reshaped_3d = tensor.expand_shape %k_final [[0], [1, 2]] output_shape [%batch, %n_head_kv, %head_dim_kv]
        : tensor<?x?xf32> into tensor<?x?x?xf32>
    %v_reshaped_3d = tensor.expand_shape %v_final [[0], [1, 2]] output_shape [%batch, %n_head_kv, %head_dim_kv]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    // Add seq_len=1 dimension for RoPE: [batch, n_head, head_dim] -> [batch, 1, n_head, head_dim]
    %q_reshaped_4d = tensor.expand_shape %q_reshaped_3d [[0], [1, 2], [3]] output_shape [%batch, %seq_len_1, %n_head, %head_dim]
        : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
    %k_reshaped_4d = tensor.expand_shape %k_reshaped_3d [[0], [1, 2], [3]] output_shape [%batch, %seq_len_1, %n_head_kv, %head_dim_kv]
        : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>

    // Apply RoPE to query and key (new token only).
    %q_rope_4d = util.call @position_components.rope(%q_reshaped_4d, %positions_2d, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf32>
    %k_rope_4d = util.call @position_components.rope(%k_reshaped_4d, %positions_2d, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf32>

    // Concat new K/V with cached K/V: [batch, ctx_len, ...] + [batch, 1, ...] -> [batch, ctx_len+1, ...]
    // K_full: [batch, ctx_len+1, n_head_kv, head_dim]
    %ctx_len_plus_1 = arith.addi %ctx_len, %c1 : index
    %k_full = tensor.concat dim(1) %k_cached, %k_rope_4d
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

    // V_full: [batch, ctx_len+1, n_head_kv, head_dim]
    // Note: v_reshaped_3d is [batch, n_head_kv, head_dim], need to expand to [batch, 1, n_head_kv, head_dim]
    %v_reshaped_4d = tensor.expand_shape %v_reshaped_3d [[0], [1, 2], [3]] output_shape [%batch, %seq_len_1, %n_head_kv, %head_dim_kv]
        : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
    %v_full = tensor.concat dim(1) %v_cached, %v_reshaped_4d
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

    // Compute attention: Q[batch, 1, ...] against K/V[batch, ctx_len+1, ...]
    // scale = 1/sqrt(head_dim)
    %head_dim_i32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32 = arith.sitofp %head_dim_i32 : i32 to f32
    %scale = math.rsqrt %head_dim_f32 : f32

    %attn_out_4d = util.call @attention_components.attention_gqa(%q_rope_4d, %k_full, %v_full, %scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32) -> tensor<?x?x?x?xf32>

    // Reshape attention output: [batch, 1, n_head, head_dim] -> [batch, n_embd]
    %attn_out_3d = tensor.collapse_shape %attn_out_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
    %attn_flat = tensor.collapse_shape %attn_out_3d [[0], [1, 2]] : tensor<?x?x?xf32> into tensor<?x?xf32>

    // Output projection: [batch, n_embd] @ [n_embd, n_embd] -> [batch, n_embd]
    %output_proj_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf32>
    %output_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%output_proj_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %output_proj = linalg.matmul ins(%attn_flat, %wo : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%output_proj_zero : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Conditionally add output bias.
    %output = scf.if %use_bias -> (tensor<?x?xf32>) {
      %output_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%output_proj, %bo : tensor<?x?xf32>, tensor<?xf32>) outs(%output_proj_init : tensor<?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?xf32>
      scf.yield %output_biased : tensor<?x?xf32>
    } else {
      scf.yield %output_proj : tensor<?x?xf32>
    }

    // Extract new K/V for cache update: [batch, 1, n_head_kv, head_dim] -> [batch, n_head_kv, head_dim]
    %k_new = tensor.collapse_shape %k_rope_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
    %v_new = tensor.collapse_shape %v_reshaped_4d [[0], [1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>

    util.return %output, %k_new, %v_new : tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>
  }

}
