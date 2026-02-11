// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fused concat + gather + GEMM + split + SwiGLU for MoE expert projections.
//
// Concatenates UP and GATE expert weights, gathers selected experts,
// performs matrix multiply (broadcasting input over expert dim),
// splits into UP and GATE halves, and applies SwiGLU activation.
//
// For each token t and each selected expert slot e:
//   expert_id = ids[e, t]
//   up_gate_w = concat(up_exps_w[expert_id], gate_exps_w[expert_id])
//   up_gate[:, e, t] = up_gate_w @ input[:, t]
//   result[:, e, t] = swiglu(up_gate)

module @moe_components {

  // External declaration - resolved by iree-link
  util.func private @activation_components.swiglu(
      tensor<?x?x?xf32>, tensor<?x?x?xf32>
  ) -> tensor<?x?x?xf32>

  util.func public @concat_gemm_id_silu(
      %input: tensor<?x?xf32>,          // [n_embd, n_tokens]
      %up_exps_w: tensor<?x?x?xf32>,    // [n_expert, n_ff, n_embd]
      %gate_exps_w: tensor<?x?x?xf32>,  // [n_expert, n_ff, n_embd]
      %ids: tensor<?x?xi32>             // [n_expert_used, n_tokens]
  ) -> tensor<?x?x?xf32> {              // [n_ff, n_expert_used, n_tokens]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32

    %n_embd = tensor.dim %input, %c0 : tensor<?x?xf32>
    %n_tokens = tensor.dim %input, %c1 : tensor<?x?xf32>
    %n_ff = tensor.dim %up_exps_w, %c1 : tensor<?x?x?xf32>
    %n_expert_used = tensor.dim %ids, %c0 : tensor<?x?xi32>
    %n_ff_2 = arith.muli %n_ff, %c2 : index

    // Step 1: Concat UP and GATE weights along n_ff dim.
    // [n_expert, n_ff, n_embd] + [n_expert, n_ff, n_embd] -> [n_expert, 2*n_ff, n_embd]
    %concat_w = tensor.concat dim(1) %up_exps_w, %gate_exps_w
        : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // Step 2: Gather selected experts.
    // [n_expert, 2*n_ff, n_embd] x [n_expert_used, n_tokens] -> [n_expert_used, n_tokens, 2*n_ff, n_embd]
    %gathered_init = tensor.empty(%n_expert_used, %n_tokens, %n_ff_2, %n_embd) : tensor<?x?x?x?xf32>
    %gathered = iree_linalg_ext.gather dimension_map = [0]
        ins(%concat_w, %ids : tensor<?x?x?xf32>, tensor<?x?xi32>)
        outs(%gathered_init : tensor<?x?x?x?xf32>)
        -> tensor<?x?x?x?xf32>

    // Step 3: Matmul with broadcasting over expert dim.
    // gathered[e, t, o, i] * input[i, t] -> output[o, e, t]  (reduce over i)
    %matmul_init = tensor.empty(%n_ff_2, %n_expert_used, %n_tokens) : tensor<?x?x?xf32>
    %matmul_filled = linalg.fill ins(%zero : f32) outs(%matmul_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %up_gate = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>,  // gathered[e, t, o, i]
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,           // input[i, t]
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>        // output[o, e, t]
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%gathered, %input : tensor<?x?x?x?xf32>, tensor<?x?xf32>)
      outs(%matmul_filled : tensor<?x?x?xf32>) {
    ^bb0(%w: f32, %in: f32, %acc: f32):
      %prod = arith.mulf %w, %in : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?xf32>

    // Step 4: Split into UP [0:n_ff] and GATE [n_ff:2*n_ff].
    %up = tensor.extract_slice %up_gate[0, 0, 0][%n_ff, %n_expert_used, %n_tokens][1, 1, 1]
        : tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %gate = tensor.extract_slice %up_gate[%n_ff, 0, 0][%n_ff, %n_expert_used, %n_tokens][1, 1, 1]
        : tensor<?x?x?xf32> to tensor<?x?x?xf32>

    // Step 5: SwiGLU activation: gate * silu(up).
    %activated = util.call @activation_components.swiglu(%gate, %up)
        : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    util.return %activated : tensor<?x?x?xf32>
  }

}
