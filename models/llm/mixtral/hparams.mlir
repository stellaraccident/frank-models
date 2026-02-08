// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mixtral-8x7B hyperparameters.
//
// Architecture summary:
//   vocab_size: 32000
//   block_count: 32 layers
//   embedding_length: 4096 (32 heads x 128 head_dim)
//   attention: 32 heads, 8 KV heads (4:1 GQA ratio)
//   feed_forward_length: 14336
//   experts: 8 total, top-2 routing
//
// Reference: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

module @hparams {

  util.func public @vocab_size() -> i64 {
    %v = arith.constant 32000 : i64
    util.return %v : i64
  }

  util.func public @block_count() -> i64 {
    %v = arith.constant 32 : i64
    util.return %v : i64
  }

  util.func public @embedding_length() -> i64 {
    %v = arith.constant 4096 : i64
    util.return %v : i64
  }

  util.func public @attention_head_count() -> i64 {
    %v = arith.constant 32 : i64
    util.return %v : i64
  }

  util.func public @attention_head_count_kv() -> i64 {
    %v = arith.constant 8 : i64
    util.return %v : i64
  }

  util.func public @feed_forward_length() -> i64 {
    %v = arith.constant 14336 : i64
    util.return %v : i64
  }

  util.func public @expert_count() -> i64 {
    %v = arith.constant 8 : i64
    util.return %v : i64
  }

  util.func public @expert_used_count() -> i64 {
    %v = arith.constant 2 : i64
    util.return %v : i64
  }

  util.func public @rope_freq_base() -> f32 {
    %v = arith.constant 10000.0 : f32
    util.return %v : f32
  }

  util.func public @layer_norm_rms_epsilon() -> f32 {
    %v = arith.constant 1.0e-5 : f32
    util.return %v : f32
  }

  // Architecture variant flags (Mixtral: no bias, no weight normalization)
  util.func public @use_attention_bias() -> i1 {
    %v = arith.constant false
    util.return %v : i1
  }

  util.func public @normalize_expert_weights() -> i1 {
    %v = arith.constant false
    util.return %v : i1
  }

}
