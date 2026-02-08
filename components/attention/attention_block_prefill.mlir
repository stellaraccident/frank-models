// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prefill Attention Block: QKV projection, RoPE, GQA attention, output projection.
// Returns K and V (with RoPE applied to K) for cache storage.
//
// This is the prefill variant that processes a full input sequence and returns
// the K/V tensors needed for KV cache population. Used during prompt processing.
//
// Logical shapes:
//   input:     [batch, seq_len, n_embd]      - input hidden states
//   positions: [batch, seq_len]              - position indices for RoPE
//   wq:        [n_embd, n_embd]              - query projection weights
//   wk:        [n_embd, n_kv_embd]           - key projection (n_kv_embd = n_head_kv * head_dim)
//   wv:        [n_embd, n_kv_embd]           - value projection
//   wo:        [n_embd, n_embd]              - output projection
//   bq/bk/bv/bo: [n_embd] or [n_kv_embd]     - optional biases
//
// Returns:
//   output:    [batch, seq_len, n_embd]      - attention output
//   k_out:     [batch, seq_len, n_head_kv, head_dim] - K with RoPE applied (for cache)
//   v_out:     [batch, seq_len, n_head_kv, head_dim] - V (for cache)

module @attention_block_prefill_components {

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

  util.func public @attention_block_prefill(
      %input: tensor<?x?x?xf32>,        // [batch, seq_len, n_embd]
      %positions: tensor<?x?xi64>,       // [batch, seq_len]
      %wq: tensor<?x?xf32>,              // [n_embd, n_embd]
      %wk: tensor<?x?xf32>,              // [n_embd, n_embd_kv]
      %wv: tensor<?x?xf32>,              // [n_embd, n_embd_kv]
      %wo: tensor<?x?xf32>,              // [n_embd, n_embd]
      %bq: tensor<?xf32>,                // [n_embd] - may be dummy if not used
      %bk: tensor<?xf32>,                // [n_embd_kv]
      %bv: tensor<?xf32>,                // [n_embd_kv]
      %bo: tensor<?xf32>,                // [n_embd]
      %use_bias: i1,                     // flag to enable/disable biases
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %rope_freq_base: f32,
      %rope_freq_scale: f32
  ) -> (tensor<?x?x?xf32>,               // output: [batch, seq_len, n_embd]
        tensor<?x?x?x?xf32>,             // k_out: [batch, seq_len, n_head_kv, head_dim]
        tensor<?x?x?x?xf32>) {           // v_out: [batch, seq_len, n_head_kv, head_dim]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?x?xf32>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?xf32>

    // Get dimensions from weights.
    %n_embd_kv = tensor.dim %wk, %c1 : tensor<?x?xf32>
    %wq_k = tensor.dim %wq, %c0 : tensor<?x?xf32>
    %wk_k = tensor.dim %wk, %c0 : tensor<?x?xf32>
    %wv_k = tensor.dim %wv, %c0 : tensor<?x?xf32>
    %wo_k = tensor.dim %wo, %c0 : tensor<?x?xf32>

    // Broadcast weights from [K, N] to [batch, K, N] for batch_matmul.
    // Q weights: [n_embd, n_embd] -> [batch, n_embd, n_embd]
    %wq_3d_init = tensor.empty(%batch, %wq_k, %n_embd) : tensor<?x?x?xf32>
    %wq_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,     // input: [K, N]
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>  // output: [batch, K, N]
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wq : tensor<?x?xf32>) outs(%wq_3d_init : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>

    // K weights: [n_embd, n_embd_kv] -> [batch, n_embd, n_embd_kv]
    %wk_3d_init = tensor.empty(%batch, %wk_k, %n_embd_kv) : tensor<?x?x?xf32>
    %wk_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wk : tensor<?x?xf32>) outs(%wk_3d_init : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>

    // V weights: [n_embd, n_embd_kv] -> [batch, n_embd, n_embd_kv]
    %wv_3d_init = tensor.empty(%batch, %wv_k, %n_embd_kv) : tensor<?x?x?xf32>
    %wv_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wv : tensor<?x?xf32>) outs(%wv_3d_init : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>

    // QKV projections: [batch, seq, n_embd] @ [batch, n_embd, n_out] -> [batch, seq, n_out]
    %cst_zero = arith.constant 0.0 : f32
    %q_proj_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf32>
    %q_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%q_proj_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %q_proj = linalg.batch_matmul ins(%input, %wq_3d : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%q_proj_zero : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    %k_proj_init = tensor.empty(%batch, %seq_len, %n_embd_kv) : tensor<?x?x?xf32>
    %k_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%k_proj_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %k_proj = linalg.batch_matmul ins(%input, %wk_3d : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%k_proj_zero : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    %v_proj_init = tensor.empty(%batch, %seq_len, %n_embd_kv) : tensor<?x?x?xf32>
    %v_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%v_proj_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %v_proj = linalg.batch_matmul ins(%input, %wv_3d : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%v_proj_zero : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // Conditionally add biases if enabled.
    %q_final = scf.if %use_bias -> (tensor<?x?x?xf32>) {
      %q_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,  // proj
          affine_map<(d0, d1, d2) -> (d2)>,          // bias
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>   // out
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%q_proj, %bq : tensor<?x?x?xf32>, tensor<?xf32>) outs(%q_proj_init : tensor<?x?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?x?xf32>
      scf.yield %q_biased : tensor<?x?x?xf32>
    } else {
      scf.yield %q_proj : tensor<?x?x?xf32>
    }

    %k_final = scf.if %use_bias -> (tensor<?x?x?xf32>) {
      %k_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%k_proj, %bk : tensor<?x?x?xf32>, tensor<?xf32>) outs(%k_proj_init : tensor<?x?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?x?xf32>
      scf.yield %k_biased : tensor<?x?x?xf32>
    } else {
      scf.yield %k_proj : tensor<?x?x?xf32>
    }

    %v_final = scf.if %use_bias -> (tensor<?x?x?xf32>) {
      %v_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%v_proj, %bv : tensor<?x?x?xf32>, tensor<?xf32>) outs(%v_proj_init : tensor<?x?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?x?xf32>
      scf.yield %v_biased : tensor<?x?x?xf32>
    } else {
      scf.yield %v_proj : tensor<?x?x?xf32>
    }

    // Reshape for multi-head: [batch, seq_len, n_embd] -> [batch, seq_len, n_head, head_dim]
    %head_dim = arith.divsi %n_embd, %n_head : index
    %head_dim_kv = arith.divsi %n_embd_kv, %n_head_kv : index

    %q_reshaped = tensor.expand_shape %q_final [[0], [1], [2, 3]] output_shape [%batch, %seq_len, %n_head, %head_dim]
        : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
    %k_reshaped = tensor.expand_shape %k_final [[0], [1], [2, 3]] output_shape [%batch, %seq_len, %n_head_kv, %head_dim_kv]
        : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
    %v_reshaped = tensor.expand_shape %v_final [[0], [1], [2, 3]] output_shape [%batch, %seq_len, %n_head_kv, %head_dim_kv]
        : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>

    // Apply RoPE to query and key.
    // K with RoPE will be stored in cache.
    %q_rope = util.call @position_components.rope(%q_reshaped, %positions, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf32>
    %k_rope = util.call @position_components.rope(%k_reshaped, %positions, %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf32>

    // Compute attention with GQA.
    // scale = 1/sqrt(head_dim)
    %head_dim_i32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32 = arith.sitofp %head_dim_i32 : i32 to f32
    %scale = math.rsqrt %head_dim_f32 : f32

    %attn_out = util.call @attention_components.attention_gqa(%q_rope, %k_rope, %v_reshaped, %scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32) -> tensor<?x?x?x?xf32>

    // Reshape back: [batch, seq_len, n_head, head_dim] -> [batch, seq_len, n_embd]
    %attn_flat = tensor.collapse_shape %attn_out [[0], [1], [2, 3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>

    // Output projection.
    %wo_3d_init = tensor.empty(%batch, %wo_k, %n_embd) : tensor<?x?x?xf32>
    %wo_3d = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%wo : tensor<?x?xf32>) outs(%wo_3d_init : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>

    %output_proj_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf32>
    %output_proj_zero = linalg.fill ins(%cst_zero : f32) outs(%output_proj_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %output_proj = linalg.batch_matmul ins(%attn_flat, %wo_3d : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%output_proj_zero : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // Conditionally add output bias.
    %output = scf.if %use_bias -> (tensor<?x?x?xf32>) {
      %output_biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%output_proj, %bo : tensor<?x?x?xf32>, tensor<?xf32>) outs(%output_proj_init : tensor<?x?x?xf32>) {
      ^bb0(%proj: f32, %bias: f32, %out: f32):
        %sum = arith.addf %proj, %bias : f32
        linalg.yield %sum : f32
      } -> tensor<?x?x?xf32>
      scf.yield %output_biased : tensor<?x?x?xf32>
    } else {
      scf.yield %output_proj : tensor<?x?x?xf32>
    }

    // Return output, K (with RoPE), V (without RoPE - applied during attention)
    // K_rope is stored in cache; V is stored as-is
    util.return %output, %k_rope, %v_reshaped : tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
  }

}
