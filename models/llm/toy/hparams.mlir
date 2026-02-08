// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Toy MoE LLM hyperparameters for fast unit testing.
//
// Dimensions chosen to be small enough for fast compilation and testing,
// but large enough to exercise all architecture patterns (GQA, MoE routing, etc.).
//
// Architecture summary:
//   vocab_size: 256
//   block_count: 2 layers
//   embedding_length: 64 (4 heads x 16 head_dim)
//   attention: 4 heads, 2 KV heads (2:1 GQA ratio)
//   feed_forward_length: 128 (2x expansion)
//   experts: 4 total, top-2 routing

module @hparams {

  util.func public @vocab_size() -> i64 {
    %v = arith.constant 256 : i64
    util.return %v : i64
  }

  util.func public @block_count() -> i64 {
    %v = arith.constant 2 : i64
    util.return %v : i64
  }

  util.func public @embedding_length() -> i64 {
    %v = arith.constant 64 : i64
    util.return %v : i64
  }

  util.func public @attention_head_count() -> i64 {
    %v = arith.constant 4 : i64
    util.return %v : i64
  }

  util.func public @attention_head_count_kv() -> i64 {
    %v = arith.constant 2 : i64
    util.return %v : i64
  }

  util.func public @feed_forward_length() -> i64 {
    %v = arith.constant 128 : i64
    util.return %v : i64
  }

  util.func public @expert_count() -> i64 {
    %v = arith.constant 4 : i64
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

  // Architecture variant flags (Mixtral-style: no bias, no weight normalization)
  util.func public @use_attention_bias() -> i1 {
    %v = arith.constant false
    util.return %v : i1
  }

  util.func public @normalize_expert_weights() -> i1 {
    %v = arith.constant false
    util.return %v : i1
  }

}
