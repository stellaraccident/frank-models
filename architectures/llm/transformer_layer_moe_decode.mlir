// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MoE Transformer Layer (Decode): pre-norm attention with cached K/V + MoE FFN.
// Processes a single token per sequence using cached K/V from previous tokens.
//
// Architecture (Mixtral-style):
//   input → attn_norm → gather(cache) → attention_block_decode → scatter_decode(cache)
//         → +residual → ffn_norm → moe_ffn_block → +residual → output
//
// Integrates with unified paged KV cache. All layers share the same physical block pool.
// Layer-aware metadata (block_tables, context_lens) has layer dimension sliced internally
// by gather/scatter.
//
// Logical shapes:
//   input:           [batch, n_embd]               - Single token hidden state per sequence
//   positions:       [batch]                       - Single position per sequence
//   cache:           !util.list<?>                 - Unified KV cache (K_blocks, V_blocks)
//   block_tables:    [n_layers, batch, max_blocks] - Block indirection
//   context_lens:    [n_layers, batch]             - Current context length per layer/seq
//   max_context_len: index                         - Max context for gather output shape
//   block_indices:   [batch]                       - Which block to write new K/V
//   pos_in_blocks:   [batch]                       - Position within block for new K/V
//
// Returns:
//   output:          [batch, n_embd]               - Output hidden state
//   cache_out:       !util.list<?>                 - Cache with new K/V written
//
// Reference: transformer_layer_moe.mlir, attention_block_decode.mlir, kvcache.mlir

module @transformer_layer_moe_decode_components {

  // ===== Parameter accessor imports (model_params module provides these) =====

  // Normalization weights
  util.func private @model_params.attn_norm_weight(i32) -> tensor<?xf32>
  util.func private @model_params.ffn_norm_weight(i32) -> tensor<?xf32>

  // Attention projection weights
  util.func private @model_params.attn_q_weight(i32) -> tensor<?x?xf32>
  util.func private @model_params.attn_k_weight(i32) -> tensor<?x?xf32>
  util.func private @model_params.attn_v_weight(i32) -> tensor<?x?xf32>
  util.func private @model_params.attn_output_weight(i32) -> tensor<?x?xf32>

  // Attention biases (may be dummy zeros if use_bias=false)
  util.func private @model_params.attn_q_bias(i32) -> tensor<?xf32>
  util.func private @model_params.attn_k_bias(i32) -> tensor<?xf32>
  util.func private @model_params.attn_v_bias(i32) -> tensor<?xf32>
  util.func private @model_params.attn_output_bias(i32) -> tensor<?xf32>

  // MoE weights
  util.func private @model_params.ffn_gate_inp_weight(i32) -> tensor<?x?xf32>
  util.func private @model_params.ffn_up_exps_weight(i32) -> tensor<?x?x?xf32>
  util.func private @model_params.ffn_gate_exps_weight(i32) -> tensor<?x?x?xf32>
  util.func private @model_params.ffn_down_exps_weight(i32) -> tensor<?x?x?xf32>

  // ===== Component imports (resolved by iree-link) =====

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf32>,    // [n_tokens, hidden_dim]
      tensor<?xf32>,       // [hidden_dim]
      f32                  // epsilon
  ) -> tensor<?x?xf32>

  // KV cache: gather K/V for a specific layer
  util.func private @kvcache_components.gather(
      !util.list<?>,           // cache
      index,                   // layer
      tensor<?x?x?xi32>,       // block_tables [n_layers, batch, max_blocks]
      tensor<?x?xi32>,         // context_lens [n_layers, batch]
      index                    // max_context_len
  ) -> (tensor<?x?x?x?xf32>,   // k_gathered: [batch, max_ctx, n_head_kv, head_dim]
        tensor<?x?x?x?xf32>)   // v_gathered: [batch, max_ctx, n_head_kv, head_dim]

  // KV cache: scatter new K/V for a specific layer (decode: single token)
  util.func private @kvcache_components.scatter_decode(
      !util.list<?>,           // cache
      index,                   // layer
      tensor<?x?x?xf32>,       // new_k: [batch, n_head_kv, head_dim]
      tensor<?x?x?xf32>,       // new_v: [batch, n_head_kv, head_dim]
      tensor<?xi32>,           // block_indices: [batch]
      tensor<?xi32>            // pos_in_blocks: [batch]
  ) -> !util.list<?>

  // Decode attention: process single token with cached K/V
  util.func private @attention_block_decode_components.attention_block_decode(
      tensor<?x?xf32>,         // [batch, n_embd]
      tensor<?xi64>,           // [batch]
      tensor<?x?x?x?xf32>,     // k_cached: [batch, ctx_len, n_head_kv, head_dim]
      tensor<?x?x?x?xf32>,     // v_cached: [batch, ctx_len, n_head_kv, head_dim]
      tensor<?x?xf32>,         // wq
      tensor<?x?xf32>,         // wk
      tensor<?x?xf32>,         // wv
      tensor<?x?xf32>,         // wo
      tensor<?xf32>,           // bq
      tensor<?xf32>,           // bk
      tensor<?xf32>,           // bv
      tensor<?xf32>,           // bo
      i1,                      // use_bias
      index,                   // n_head
      index,                   // n_head_kv
      index,                   // n_embd
      f32,                     // rope_freq_base
      f32                      // rope_freq_scale
  ) -> (tensor<?x?xf32>,       // output: [batch, n_embd]
        tensor<?x?x?xf32>,     // k_new: [batch, n_head_kv, head_dim]
        tensor<?x?x?xf32>)     // v_new: [batch, n_head_kv, head_dim]

  util.func private @moe_ffn_components.moe_ffn_block(
      tensor<?x?xf32>,       // [n_tokens, n_embd]
      tensor<?x?xf32>,       // gate_inp_weight
      tensor<?x?x?xf32>,     // up_exps_weight
      tensor<?x?x?xf32>,     // gate_exps_weight
      tensor<?x?x?xf32>,     // down_exps_weight
      index,                 // n_expert
      index,                 // n_expert_used
      index,                 // n_embd
      index,                 // n_ff
      i1                     // normalize_weights
  ) -> tensor<?x?xf32>

  // ===== Layer function =====

  util.func public @transformer_layer_moe_decode(
      %input: tensor<?x?xf32>,             // [batch, n_embd]
      %positions: tensor<?xi64>,            // [batch]
      %cache: !util.list<?>,                // Unified KV cache
      %block_tables: tensor<?x?x?xi32>,     // [n_layers, batch, max_blocks]
      %context_lens: tensor<?x?xi32>,       // [n_layers, batch]
      %max_context_len: index,
      %block_indices: tensor<?xi32>,        // [batch] - which block for new K/V
      %pos_in_blocks: tensor<?xi32>,        // [batch] - position within block
      %layer_idx: i32,
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %n_ff: index,
      %n_expert: index,
      %n_expert_used: index,
      %rms_eps: f32,
      %rope_freq_base: f32,
      %rope_freq_scale: f32,
      %use_bias: i1,
      %normalize_weights: i1
  ) -> (tensor<?x?xf32>,                    // output: [batch, n_embd]
        !util.list<?>) {                    // cache_out with new K/V written
    %c0 = arith.constant 0 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?xf32>

    // Convert layer_idx to index for kvcache calls.
    %layer = arith.index_cast %layer_idx : i32 to index

    // ---- Load all parameters for this layer ----

    %attn_norm_w = util.call @model_params.attn_norm_weight(%layer_idx) : (i32) -> tensor<?xf32>
    %ffn_norm_w = util.call @model_params.ffn_norm_weight(%layer_idx) : (i32) -> tensor<?xf32>

    %wq = util.call @model_params.attn_q_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %wk = util.call @model_params.attn_k_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %wv = util.call @model_params.attn_v_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %wo = util.call @model_params.attn_output_weight(%layer_idx) : (i32) -> tensor<?x?xf32>

    %bq = util.call @model_params.attn_q_bias(%layer_idx) : (i32) -> tensor<?xf32>
    %bk = util.call @model_params.attn_k_bias(%layer_idx) : (i32) -> tensor<?xf32>
    %bv = util.call @model_params.attn_v_bias(%layer_idx) : (i32) -> tensor<?xf32>
    %bo = util.call @model_params.attn_output_bias(%layer_idx) : (i32) -> tensor<?xf32>

    %gate_inp_w = util.call @model_params.ffn_gate_inp_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %up_exps_w = util.call @model_params.ffn_up_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf32>
    %gate_exps_w = util.call @model_params.ffn_gate_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf32>
    %down_exps_w = util.call @model_params.ffn_down_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf32>

    // ---- Attention sub-layer ----

    // RMS norm on input: [batch, n_embd].
    %attn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %input, %attn_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    // Gather cached K/V for this layer: [batch, max_ctx, n_head_kv, head_dim].
    %k_cached, %v_cached = util.call @kvcache_components.gather(
        %cache, %layer, %block_tables, %context_lens, %max_context_len)
        : (!util.list<?>, index, tensor<?x?x?xi32>, tensor<?x?xi32>, index)
        -> (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)

    // Decode attention with cached K/V.
    %attn_out, %k_new, %v_new = util.call @attention_block_decode_components.attention_block_decode(
        %attn_normed, %positions,
        %k_cached, %v_cached,
        %wq, %wk, %wv, %wo,
        %bq, %bk, %bv, %bo,
        %use_bias, %n_head, %n_head_kv, %n_embd,
        %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?xf32>, tensor<?xi64>,
           tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>,
           tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
           tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
           i1, index, index, index, f32, f32)
        -> (tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)

    // Scatter new K/V to cache.
    %cache_updated = util.call @kvcache_components.scatter_decode(
        %cache, %layer, %k_new, %v_new, %block_indices, %pos_in_blocks)
        : (!util.list<?>, index, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
           tensor<?xi32>, tensor<?xi32>) -> !util.list<?>

    // Residual connection: input + attn_out.
    %residual1_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf32>
    %residual1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %attn_out : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%residual1_init : tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>

    // ---- MoE FFN sub-layer ----

    // RMS norm on residual: [batch, n_embd].
    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %residual1, %ffn_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    // MoE FFN block operates on [batch, n_embd] (batch = n_tokens for decode).
    %moe_out = util.call @moe_ffn_components.moe_ffn_block(
        %ffn_normed, %gate_inp_w,
        %up_exps_w, %gate_exps_w, %down_exps_w,
        %n_expert, %n_expert_used, %n_embd, %n_ff,
        %normalize_weights)
        : (tensor<?x?xf32>, tensor<?x?xf32>,
           tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
           index, index, index, index, i1) -> tensor<?x?xf32>

    // Residual connection: residual1 + moe_out.
    %output_init = tensor.empty(%batch, %n_embd) : tensor<?x?xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%residual1, %moe_out : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%output_init : tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>

    util.return %output, %cache_updated : tensor<?x?xf32>, !util.list<?>
  }

}
