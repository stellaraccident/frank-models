// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MoE Transformer Layer: pre-norm attention + MoE FFN with residual connections.
//
// Architecture (Mixtral-style):
//   input → attn_norm → attention_block → +residual
//         → ffn_norm  → moe_ffn_block  → +residual → output
//
// Parameters accessed via @model_params.* accessor functions (provided by specialized module).
// Computation delegated to component imports (resolved by iree-link).
//
// Reference: kb/ben/moe_f32_parameterized.mlir @transformer_layer_moe

module @transformer_layer_moe_components {

  // ===== Parameter accessor imports (model_params module provides these) =====
  // Each takes a layer index and returns the parameter tensor for that layer.
  // Dependency injection pattern: layer imports from model_params namespace.

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
      tensor<?x?xf32>, tensor<?xf32>, f32
  ) -> tensor<?x?xf32>

  util.func private @attention_block_components.attention_block(
      tensor<?x?x?xf32>, tensor<?x?xi64>,
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
      tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
      i1, index, index, index, f32, f32
  ) -> tensor<?x?x?xf32>

  util.func private @moe_ffn_components.moe_ffn_block(
      tensor<?x?xf32>, tensor<?x?xf32>,
      tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
      index, index, index, index, i1
  ) -> tensor<?x?xf32>

  // ===== Layer function =====

  util.func public @transformer_layer_moe(
      %input: tensor<?x?x?xf32>,      // [batch, seq_len, n_embd]
      %positions: tensor<?x?xi64>,     // [batch, seq_len]
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
  ) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?x?xf32>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?xf32>
    %n_tokens = arith.muli %batch, %seq_len : index

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

    // Flatten [batch, seq_len, n_embd] → [batch*seq_len, n_embd] for rms_norm (2D).
    %input_2d = tensor.collapse_shape %input [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>

    %attn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %input_2d, %attn_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    // Unflatten back to [batch, seq_len, n_embd] for attention_block (3D).
    %attn_normed_3d = tensor.expand_shape %attn_normed [[0, 1], [2]]
        output_shape [%batch, %seq_len, %n_embd]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    %attn_out = util.call @attention_block_components.attention_block(
        %attn_normed_3d, %positions,
        %wq, %wk, %wv, %wo,
        %bq, %bk, %bv, %bo,
        %use_bias, %n_head, %n_head_kv, %n_embd,
        %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?xf32>, tensor<?x?xi64>,
           tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
           tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
           i1, index, index, index, f32, f32) -> tensor<?x?x?xf32>

    // Residual connection: input + attn_out.
    %residual1_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf32>
    %residual1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %attn_out : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%residual1_init : tensor<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?xf32>

    // ---- MoE FFN sub-layer ----

    // Flatten [batch, seq_len, n_embd] → [n_tokens, n_embd] for rms_norm + moe_ffn_block.
    %residual1_2d = tensor.collapse_shape %residual1 [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>

    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %residual1_2d, %ffn_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    // MoE FFN block operates on [n_tokens, n_embd].
    %moe_out = util.call @moe_ffn_components.moe_ffn_block(
        %ffn_normed, %gate_inp_w,
        %up_exps_w, %gate_exps_w, %down_exps_w,
        %n_expert, %n_expert_used, %n_embd, %n_ff,
        %normalize_weights)
        : (tensor<?x?xf32>, tensor<?x?xf32>,
           tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
           index, index, index, index, i1) -> tensor<?x?xf32>

    // Unflatten MoE output back to [batch, seq_len, n_embd].
    // NOTE: This expand_shape currently triggers a dimension inference error in full model
    // compilation (PropagateLinalgTransposePass). The compiler cannot trace that
    // dim(%moe_out, 0) == %batch * %seq_len through the moe_ffn_block function call,
    // even though moe_ffn_block explicitly preserves n_tokens dimension.
    // Works fine in isolated layer tests but fails when inlined in full forward pass.
    %moe_out_3d = tensor.expand_shape %moe_out [[0, 1], [2]]
        output_shape [%batch, %seq_len, %n_embd]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    // Residual connection: residual1 + moe_out.
    %output_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%residual1, %moe_out_3d : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%output_init : tensor<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?xf32>

    util.return %output : tensor<?x?x?xf32>
  }

}
