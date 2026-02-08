// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mixtral-8x7B with KV Cache integration.
// Separate @prefill and @decode entry points for efficient autoregressive generation.
//
// Architecture: 32 layers, 4096 embd, 32 heads, 8 kv_heads, 14336 ff, 8 experts (top-2).
// Unified paged KV cache: single block pool shared across all layers.
// Layer-aware metadata: block_tables[n_layers, batch, max_blocks], context_lens[n_layers, batch].
//
// Entry points:
//   @allocate_kv_cache(n_blocks, block_size) -> cache  // derives n_head_kv, head_dim from hparams
//   @prefill(tokens, positions, cache, ...) -> (logits, cache_out)
//   @decode(tokens, positions, cache, ...) -> (logits, cache_out)
//
// Note: Both prefill and decode integrate scatter to cache directly.
//       Cache is threaded through layer loop and returned with K/V written.
//
// Reference: mixtral.mlir, kvcache.mlir, transformer_layer_moe_prefill/decode.mlir

module @mixtral_cached {

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
      tensor<?x?xf32>,   // [vocab_size, n_embd]
      tensor<?x?xi64>    // [batch, seq_len]
  ) -> tensor<?x?x?xf32> // [batch, seq_len, n_embd]

  util.func private @embedding_components.embedding_lookup_1d(
      tensor<?x?xf32>,   // [vocab_size, n_embd]
      tensor<?xi64>      // [batch]
  ) -> tensor<?x?xf32>   // [batch, n_embd]

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf32>,   // [n_tokens, hidden_dim]
      tensor<?xf32>,      // [hidden_dim]
      f32                 // epsilon
  ) -> tensor<?x?xf32>

  // KV cache
  util.func private @kvcache_components.allocate(
      index,   // n_blocks
      index,   // block_size
      index,   // n_head_kv
      index    // head_dim
  ) -> !util.list<?>

  // Prefill transformer layer (scatters K/V to cache internally)
  util.func private @transformer_layer_moe_prefill_components.transformer_layer_moe_prefill(
      tensor<?x?x?xf32>,   // input: [batch, seq_len, n_embd]
      tensor<?x?xi64>,     // positions: [batch, seq_len]
      !util.list<?>,       // cache
      tensor<?x?x?xi32>,   // block_tables: [n_layers, batch, max_blocks]
      tensor<?xi32>,       // start_positions: [batch]
      index,               // block_size
      i32,                 // layer_idx
      index,               // n_head
      index,               // n_head_kv
      index,               // n_embd
      index,               // n_ff
      index,               // n_expert
      index,               // n_expert_used
      f32,                 // rms_eps
      f32,                 // rope_freq_base
      f32,                 // rope_freq_scale
      i1,                  // use_bias
      i1                   // normalize_weights
  ) -> (tensor<?x?x?xf32>,     // output: [batch, seq_len, n_embd]
        !util.list<?>)         // cache_out with K/V written

  // Decode transformer layer
  util.func private @transformer_layer_moe_decode_components.transformer_layer_moe_decode(
      tensor<?x?xf32>,         // input: [batch, n_embd]
      tensor<?xi64>,           // positions: [batch]
      !util.list<?>,           // cache
      tensor<?x?x?xi32>,       // block_tables: [n_layers, batch, max_blocks]
      tensor<?x?xi32>,         // context_lens: [n_layers, batch]
      index,                   // max_context_len
      tensor<?xi32>,           // block_indices: [batch]
      tensor<?xi32>,           // pos_in_blocks: [batch]
      i32,                     // layer_idx
      index,                   // n_head
      index,                   // n_head_kv
      index,                   // n_embd
      index,                   // n_ff
      index,                   // n_expert
      index,                   // n_expert_used
      f32,                     // rms_eps
      f32,                     // rope_freq_base
      f32,                     // rope_freq_scale
      i1,                      // use_bias
      i1                       // normalize_weights
  ) -> (tensor<?x?xf32>,       // output: [batch, n_embd]
        !util.list<?>)         // cache_out

  // ===== KV Cache Allocation =====
  // Derives n_head_kv and head_dim from hparams.

  util.func public @allocate_kv_cache(
      %n_blocks: index,
      %block_size: index
  ) -> !util.list<?> {
    // Get n_head_kv and head_dim from hyperparameters.
    %n_head_kv_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %n_embd_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_head_i64 = util.call @hparams.attention_head_count() : () -> i64

    %n_head_kv = arith.index_cast %n_head_kv_i64 : i64 to index
    %n_embd = arith.index_cast %n_embd_i64 : i64 to index
    %n_head = arith.index_cast %n_head_i64 : i64 to index
    %head_dim = arith.divsi %n_embd, %n_head : index

    %cache = util.call @kvcache_components.allocate(
        %n_blocks, %block_size, %n_head_kv, %head_dim)
        : (index, index, index, index) -> !util.list<?>
    util.return %cache : !util.list<?>
  }

  // ===== Prefill Entry Point =====
  // Process initial tokens, scatter K/V to cache, return logits and updated cache.
  //
  // Logical shapes:
  //   tokens:          [batch, seq_len]            - Input token IDs
  //   positions:       [batch, seq_len]            - Position indices
  //   cache:           !util.list<?>               - Unified KV cache
  //   block_tables:    [n_layers, batch, max_blocks] - Block indirection
  //   start_positions: [batch]                     - Where to start writing (usually 0)
  //   block_size:      index                       - Tokens per cache block
  //
  // Returns:
  //   logits:          [batch, seq_len, vocab_size] - Output logits for all positions
  //   cache_out:       !util.list<?>               - Cache with K/V written
  //
  // Note: Each layer scatters K/V to cache internally via scatter_prefill.
  // Assumes uniform sequence lengths (ragged scatter is a future optimization).

  util.func public @prefill(
      %tokens: tensor<?x?xi64>,           // [batch, seq_len]
      %positions: tensor<?x?xi64>,        // [batch, seq_len]
      %cache: !util.list<?>,              // Unified KV cache
      %block_tables: tensor<?x?x?xi32>,   // [n_layers, batch, max_blocks]
      %start_positions: tensor<?xi32>,    // [batch]
      %block_size: index
  ) -> (tensor<?x?x?xf32>,                // logits: [batch, seq_len, vocab_size]
        !util.list<?>) {                  // cache_out
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

    // Convert to index with ranges.
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

    %batch = tensor.dim %tokens, %c0 : tensor<?x?xi64>
    %seq_len = tensor.dim %tokens, %c1 : tensor<?x?xi64>
    %batch_assumed = util.assume.int %batch<umin = 1, umax = 32> : index
    %seq_len_assumed = util.assume.int %seq_len<umin = 1, umax = 32768> : index

    // Mixtral has no biases, no weight normalization.
    %false = arith.constant false

    // Token embedding lookup.
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf32>
    %embeddings = util.call @embedding_components.embedding_lookup(%tok_embd_weight, %tokens)
        : (tensor<?x?xf32>, tensor<?x?xi64>) -> tensor<?x?x?xf32>

    // Transformer layers loop (prefill variant with cache threading).
    // Each layer scatters K/V to cache internally via scatter_prefill.
    %rope_freq_scale = arith.constant 1.0 : f32
    %final_hidden, %final_cache = scf.for %layer_idx = %c0 to %n_layer_assumed step %c1
        iter_args(%hidden = %embeddings, %cache_iter = %cache) -> (tensor<?x?x?xf32>, !util.list<?>) {
      %layer_idx_i32 = arith.index_cast %layer_idx : index to i32

      // Prefill layer: compute attention, scatter K/V to cache, return output + updated cache.
      %layer_out, %cache_updated = util.call @transformer_layer_moe_prefill_components.transformer_layer_moe_prefill(
          %hidden, %positions, %cache_iter,
          %block_tables, %start_positions, %block_size,
          %layer_idx_i32,
          %n_head_assumed, %n_head_kv_assumed, %n_embd_assumed, %n_ff_assumed,
          %n_expert_assumed, %n_expert_used_assumed,
          %rms_eps, %rope_freq_base, %rope_freq_scale,
          %false, %false)
          : (tensor<?x?x?xf32>, tensor<?x?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, tensor<?xi32>, index,
             i32,
             index, index, index, index, index, index,
             f32, f32, f32, i1, i1)
          -> (tensor<?x?x?xf32>, !util.list<?>)

      scf.yield %layer_out, %cache_updated : tensor<?x?x?xf32>, !util.list<?>
    }

    // Output normalization.
    %final_hidden_2d = tensor.collapse_shape %final_hidden [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf32>
    %normalized_2d = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden_2d, %output_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    %normalized = tensor.expand_shape %normalized_2d [[0, 1], [2]]
        output_shape [%batch_assumed, %seq_len_assumed, %n_embd_assumed]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    // LM head projection.
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

    util.return %logits, %final_cache : tensor<?x?x?xf32>, !util.list<?>
  }

  // ===== Decode Entry Point =====
  // Process single token per sequence using cached K/V.
  //
  // Logical shapes:
  //   tokens:          [batch]                       - Single token per sequence
  //   positions:       [batch]                       - Single position per sequence
  //   cache:           !util.list<?>                 - KV cache with past K/V
  //   block_tables:    [n_layers, batch, max_blocks] - Block indirection
  //   context_lens:    [n_layers, batch]             - Current context length
  //   max_context_len: index                         - Max context for gather
  //   block_indices:   [batch]                       - Block for new K/V
  //   pos_in_blocks:   [batch]                       - Position within block
  //
  // Returns:
  //   logits:          [batch, vocab_size]           - Next-token logits
  //   cache_out:       !util.list<?>                 - Cache with new K/V written

  util.func public @decode(
      %tokens: tensor<?xi64>,               // [batch]
      %positions: tensor<?xi64>,            // [batch]
      %cache: !util.list<?>,
      %block_tables: tensor<?x?x?xi32>,     // [n_layers, batch, max_blocks]
      %context_lens: tensor<?x?xi32>,       // [n_layers, batch]
      %max_context_len: index,
      %block_indices: tensor<?xi32>,        // [batch]
      %pos_in_blocks: tensor<?xi32>         // [batch]
  ) -> (tensor<?x?xf32>,                    // logits: [batch, vocab_size]
        !util.list<?>) {                    // cache_out
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

    // Convert to index with ranges.
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

    %batch = tensor.dim %tokens, %c0 : tensor<?xi64>
    %batch_assumed = util.assume.int %batch<umin = 1, umax = 32> : index

    // Mixtral has no biases, no weight normalization.
    %false = arith.constant false

    // Token embedding lookup (1D variant for decode).
    %tok_embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf32>
    %embeddings = util.call @embedding_components.embedding_lookup_1d(%tok_embd_weight, %tokens)
        : (tensor<?x?xf32>, tensor<?xi64>) -> tensor<?x?xf32>

    // Transformer layers loop (decode variant with cache threading).
    %rope_freq_scale = arith.constant 1.0 : f32
    %final_hidden, %final_cache = scf.for %layer_idx = %c0 to %n_layer_assumed step %c1
        iter_args(%hidden = %embeddings, %cache_iter = %cache) -> (tensor<?x?xf32>, !util.list<?>) {
      %layer_idx_i32 = arith.index_cast %layer_idx : index to i32

      // Decode layer: gather cached K/V, compute attention, scatter new K/V.
      %layer_out, %cache_updated = util.call @transformer_layer_moe_decode_components.transformer_layer_moe_decode(
          %hidden, %positions, %cache_iter,
          %block_tables, %context_lens, %max_context_len,
          %block_indices, %pos_in_blocks,
          %layer_idx_i32,
          %n_head_assumed, %n_head_kv_assumed, %n_embd_assumed, %n_ff_assumed,
          %n_expert_assumed, %n_expert_used_assumed,
          %rms_eps, %rope_freq_base, %rope_freq_scale,
          %false, %false)
          : (tensor<?x?xf32>, tensor<?xi64>, !util.list<?>,
             tensor<?x?x?xi32>, tensor<?x?xi32>, index,
             tensor<?xi32>, tensor<?xi32>,
             i32,
             index, index, index, index, index, index,
             f32, f32, f32, i1, i1)
          -> (tensor<?x?xf32>, !util.list<?>)

      scf.yield %layer_out, %cache_updated : tensor<?x?xf32>, !util.list<?>
    }

    // Output normalization.
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf32>
    %normalized = util.call @rms_norm_components.rms_norm_linalg(
        %final_hidden, %output_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    // LM head projection: [batch, n_embd] @ [n_embd, vocab] -> [batch, vocab].
    %output_weight = util.call @model_params.output_weight() : () -> tensor<?x?xf32>
    %logits_empty = tensor.empty(%batch_assumed, %n_vocab_assumed) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %logits_init = linalg.fill ins(%zero : f32) outs(%logits_empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %logits = linalg.matmul ins(%normalized, %output_weight : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%logits_init : tensor<?x?xf32>) -> tensor<?x?xf32>

    util.return %logits, %final_cache : tensor<?x?xf32>, !util.list<?>
  }

}
