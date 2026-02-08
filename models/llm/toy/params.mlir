// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Toy MoE LLM parameter accessors.
//
// Uses flow.tensor.constant with #flow.parameter.named to load parameters
// from an IREE parameter archive at runtime.
//
// Parameter naming follows GGUF convention:
//   Model-level: token_embd.weight, output_norm.weight, output.weight
//   Layer-level: blk.{idx}.{name} (e.g., blk.0.attn_q.weight)
//
// Shapes (toy dimensions):
//   vocab_size=256, n_embd=64, n_head=4, n_head_kv=2, head_dim=16
//   n_ff=128, n_expert=4, n_layers=2

module @model_params {

  // ===== Model-level parameters =====

  util.func public @token_embd_weight() -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"token_embd.weight"> : tensor<256x64xf32>
    %dyn = tensor.cast %w : tensor<256x64xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @output_norm_weight() -> tensor<?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output_norm.weight"> : tensor<64xf32>
    %dyn = tensor.cast %w : tensor<64xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @output_weight() -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output.weight"> : tensor<64x256xf32>
    %dyn = tensor.cast %w : tensor<64x256xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  // ===== Layer-level parameters =====
  // Each function takes layer_idx and dispatches to the correct layer's parameter.
  // For toy model with 2 layers, we use scf.if to select between layer 0 and 1.

  // --- Normalization weights ---

  util.func public @attn_norm_weight(%layer: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_norm.weight"> : tensor<64xf32>
      scf.yield %w0 : tensor<64xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_norm.weight"> : tensor<64xf32>
      scf.yield %w1 : tensor<64xf32>
    }
    %dyn = tensor.cast %w : tensor<64xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @ffn_norm_weight(%layer: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_norm.weight"> : tensor<64xf32>
      scf.yield %w0 : tensor<64xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_norm.weight"> : tensor<64xf32>
      scf.yield %w1 : tensor<64xf32>
    }
    %dyn = tensor.cast %w : tensor<64xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  // --- Attention projection weights ---

  util.func public @attn_q_weight(%layer: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64x64xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_q.weight"> : tensor<64x64xf32>
      scf.yield %w0 : tensor<64x64xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_q.weight"> : tensor<64x64xf32>
      scf.yield %w1 : tensor<64x64xf32>
    }
    %dyn = tensor.cast %w : tensor<64x64xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @attn_k_weight(%layer: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64x32xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_k.weight"> : tensor<64x32xf32>
      scf.yield %w0 : tensor<64x32xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_k.weight"> : tensor<64x32xf32>
      scf.yield %w1 : tensor<64x32xf32>
    }
    %dyn = tensor.cast %w : tensor<64x32xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @attn_v_weight(%layer: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64x32xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_v.weight"> : tensor<64x32xf32>
      scf.yield %w0 : tensor<64x32xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_v.weight"> : tensor<64x32xf32>
      scf.yield %w1 : tensor<64x32xf32>
    }
    %dyn = tensor.cast %w : tensor<64x32xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @attn_output_weight(%layer: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64x64xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_output.weight"> : tensor<64x64xf32>
      scf.yield %w0 : tensor<64x64xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_output.weight"> : tensor<64x64xf32>
      scf.yield %w1 : tensor<64x64xf32>
    }
    %dyn = tensor.cast %w : tensor<64x64xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  // --- Attention biases (zeros since use_bias=false, but interface requires them) ---

  util.func public @attn_q_bias(%layer: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_q.bias"> : tensor<64xf32>
      scf.yield %w0 : tensor<64xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_q.bias"> : tensor<64xf32>
      scf.yield %w1 : tensor<64xf32>
    }
    %dyn = tensor.cast %w : tensor<64xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @attn_k_bias(%layer: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<32xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_k.bias"> : tensor<32xf32>
      scf.yield %w0 : tensor<32xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_k.bias"> : tensor<32xf32>
      scf.yield %w1 : tensor<32xf32>
    }
    %dyn = tensor.cast %w : tensor<32xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @attn_v_bias(%layer: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<32xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_v.bias"> : tensor<32xf32>
      scf.yield %w0 : tensor<32xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_v.bias"> : tensor<32xf32>
      scf.yield %w1 : tensor<32xf32>
    }
    %dyn = tensor.cast %w : tensor<32xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @attn_output_bias(%layer: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_output.bias"> : tensor<64xf32>
      scf.yield %w0 : tensor<64xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_output.bias"> : tensor<64xf32>
      scf.yield %w1 : tensor<64xf32>
    }
    %dyn = tensor.cast %w : tensor<64xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  // --- MoE weights ---

  util.func public @ffn_gate_inp_weight(%layer: i32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<4x64xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_gate_inp.weight"> : tensor<4x64xf32>
      scf.yield %w0 : tensor<4x64xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_gate_inp.weight"> : tensor<4x64xf32>
      scf.yield %w1 : tensor<4x64xf32>
    }
    %dyn = tensor.cast %w : tensor<4x64xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @ffn_up_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<128x64x4xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_up_exps.weight"> : tensor<128x64x4xf32>
      scf.yield %w0 : tensor<128x64x4xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_up_exps.weight"> : tensor<128x64x4xf32>
      scf.yield %w1 : tensor<128x64x4xf32>
    }
    %dyn = tensor.cast %w : tensor<128x64x4xf32> to tensor<?x?x?xf32>
    util.return %dyn : tensor<?x?x?xf32>
  }

  util.func public @ffn_gate_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<128x64x4xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_gate_exps.weight"> : tensor<128x64x4xf32>
      scf.yield %w0 : tensor<128x64x4xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_gate_exps.weight"> : tensor<128x64x4xf32>
      scf.yield %w1 : tensor<128x64x4xf32>
    }
    %dyn = tensor.cast %w : tensor<128x64x4xf32> to tensor<?x?x?xf32>
    util.return %dyn : tensor<?x?x?xf32>
  }

  util.func public @ffn_down_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : i32
    %is_layer0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is_layer0 -> (tensor<64x128x4xf32>) {
      %w0 = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_down_exps.weight"> : tensor<64x128x4xf32>
      scf.yield %w0 : tensor<64x128x4xf32>
    } else {
      %w1 = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.ffn_down_exps.weight"> : tensor<64x128x4xf32>
      scf.yield %w1 : tensor<64x128x4xf32>
    }
    %dyn = tensor.cast %w : tensor<64x128x4xf32> to tensor<?x?x?xf32>
    util.return %dyn : tensor<?x?x?xf32>
  }

}
