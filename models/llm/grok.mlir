// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Grok-1 forward pass.
// Reference: llama.cpp llama-model.cpp:7215-7382 (llm_build_grok).
//
// Architecture: 64 layers, 6144 embd, 48 heads, 8 kv_heads, 32768 ff, 8 experts (top-2).
// No biases on attention or FFN.
// No expert weight normalization.
// Optional shared FFN (can coexist with MoE) - not yet implemented.

module @grok {

  // ===== Hyperparameter imports (hparams module provides these) =====
  util.func private @hparams.vocab_size() -> i64
  util.func private @hparams.block_count() -> i64
  util.func private @hparams.embedding_length() -> i64
  util.func private @hparams.attention_head_count() -> i64
  util.func private @hparams.attention_head_count_kv() -> i64
  util.func private @hparams.feed_forward_length() -> i64
  util.func private @hparams.expert_count() -> i64
  util.func private @hparams.expert_used_count() -> i64
  util.func private @hparams.rope_freq_base() -> f32
  util.func private @hparams.layer_norm_rms_epsilon() -> f32

  // ===== Model-level parameter imports (model_params module provides these) =====
  util.func private @model_params.token_embd_weight() -> tensor<?x?xf32>
  util.func private @model_params.output_norm_weight() -> tensor<?xf32>
  util.func private @model_params.output_weight() -> tensor<?x?xf32>

  // ===== Component imports (resolved by iree-link) =====
  util.func private @embedding_components.embedding_lookup(
      tensor<?x?xf32>, tensor<?x?xi64>
  ) -> tensor<?x?x?xf32>

  util.func private @transformer_layer_moe_components.transformer_layer_moe(
      tensor<?x?x?xf32>, tensor<?x?xi64>, i32,
      index, index, index, index, index, index,
      f32, f32, f32, i1, i1
  ) -> tensor<?x?x?xf32>

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf32>, tensor<?xf32>, f32
  ) -> tensor<?x?xf32>

  // ===== Forward pass =====

  util.func public @forward(
      %tokens: tensor<?x?xi64>,         // Input token IDs [batch, seq_len].
      %positions: tensor<?x?xi64>       // Position indices [batch, seq_len].
  ) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Load hyperparameters.
    %n_vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layer_i64 = util.call @hparams.block_count() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_ff_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %n_expert_i64 = util.call @hparams.expert_count() : () -> i64
    %n_expert_used_i64 = util.call @hparams.expert_used_count() : () -> i64
    %rope_freq_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    // Convert to index type with BROAD ranges for multiple MoE architectures.
    %n_vocab = arith.index_cast %n_vocab_i64 : i64 to index
    %n_vocab_assumed = util.assume.int %n_vocab<umin = 32000, umax = 200000> : index

    %n_layer = arith.index_cast %n_layer_i64 : i64 to index
    %n_layer_assumed = util.assume.int %n_layer<umin = 16, umax = 80> : index

    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_embd_assumed = util.assume.int %n_embd<umin = 2048, umax = 8192> : index

    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %n_head_assumed = util.assume.int %n_head<umin = 16, umax = 64> : index

    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_head_kv_assumed = util.assume.int %n_head_kv<umin = 4, umax = 16> : index

    %n_ff = arith.index_cast %n_ff_i64 : i64 to index
    %n_ff_assumed = util.assume.int %n_ff<umin = 8192, umax = 32768> : index

    %n_expert = arith.index_cast %n_expert_i64 : i64 to index
    %n_expert_assumed = util.assume.int %n_expert<umin = 4, umax = 64> : index

    %n_expert_used = arith.index_cast %n_expert_used_i64 : i64 to index
    %n_expert_used_assumed = util.assume.int %n_expert_used<umin = 1, umax = 8> : index

    // Batch and sequence length with broad ranges.
    %batch = tensor.dim %tokens, %c0 : tensor<?x?xi64>
    %seq_len = tensor.dim %tokens, %c1 : tensor<?x?xi64>
    %batch_assumed = util.assume.int %batch<umin = 1, umax = 32> : index
    %seq_len_assumed = util.assume.int %seq_len<umin = 1, umax = 32768> : index

    // Grok has no biases.
    %false = arith.constant false

    // Grok does not normalize expert weights.
    %normalize_weights = arith.constant false

    // Token embedding lookup.
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf32>
    %embeddings = util.call @embedding_components.embedding_lookup(%tok_embd_weight, %tokens)
        : (tensor<?x?xf32>, tensor<?x?xi64>) -> tensor<?x?x?xf32>

    // Transformer layers loop.
    %rope_freq_scale = arith.constant 1.0 : f32
    %final_hidden = scf.for %layer_idx = %c0 to %n_layer_assumed step %c1
        iter_args(%hidden = %embeddings) -> (tensor<?x?x?xf32>) {
      %layer_idx_i32 = arith.index_cast %layer_idx : index to i32
      %layer_out = util.call @transformer_layer_moe_components.transformer_layer_moe(
          %hidden, %positions, %layer_idx_i32,
          %n_head_assumed, %n_head_kv_assumed, %n_embd_assumed, %n_ff_assumed,
          %n_expert_assumed, %n_expert_used_assumed,
          %rope_freq_base, %rope_freq_scale, %rms_eps,
          %false, %normalize_weights)
          : (tensor<?x?x?xf32>, tensor<?x?xi64>, i32, index, index, index,
             index, index, index, f32, f32, f32, i1, i1) -> tensor<?x?x?xf32>

      scf.yield %layer_out : tensor<?x?x?xf32>
    }

    // Output normalization (flatten to 2D for rms_norm).
    %final_hidden_2d = tensor.collapse_shape %final_hidden [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf32>
    %normalized_2d = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden_2d, %output_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    // Unflatten back to 3D for batch_matmul.
    %normalized = tensor.expand_shape %normalized_2d [[0, 1], [2]]
        output_shape [%batch_assumed, %seq_len_assumed, %n_embd_assumed]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    // LM head projection (no bias).
    // Expand output_weight from [n_embd, n_vocab] to [1, n_embd, n_vocab] for batch_matmul.
    %output_weight_2d = util.call @model_params.output_weight() : () -> tensor<?x?xf32>
    %output_weight_static = tensor.expand_shape %output_weight_2d [[0, 1], [2]]
        output_shape [%c1, %n_embd_assumed, %n_vocab_assumed]
        : tensor<?x?xf32> into tensor<1x?x?xf32>
    %output_weight = tensor.cast %output_weight_static : tensor<1x?x?xf32> to tensor<?x?x?xf32>
    %logits_empty = tensor.empty(%batch_assumed, %seq_len_assumed, %n_vocab_assumed) : tensor<?x?x?xf32>
    %zero = arith.constant 0.0 : f32
    %logits_init = linalg.fill ins(%zero : f32) outs(%logits_empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %logits = linalg.batch_matmul ins(%normalized, %output_weight : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%logits_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    util.return %logits : tensor<?x?x?xf32>
  }

}
