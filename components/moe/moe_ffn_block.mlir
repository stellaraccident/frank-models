// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MoE FFN block with expert routing and mixture.
// Reference: llama.cpp llama-graph.cpp:820-1084 (build_moe_ffn).
//
// Complete MoE flow:
// 1. Router logits via gating network
// 2. Softmax gating function
// 3. Top-k expert selection
// 4. Extract and optionally normalize expert weights
// 5. Fused UP + GATE projection with SwiGLU via concat_gemm_id_silu
// 6. Expert DOWN projection (gather + matmul)
// 7. Apply expert weights element-wise
// 8. Sum expert outputs
//
// Each token independently routes to its top-k experts, enabling efficient batching.

module @moe_ffn_components {

  // External declaration - resolved by iree-link
  util.func private @moe_components.concat_gemm_id_silu(
      tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xi32>
  ) -> tensor<?x?x?xf32>

  util.func public @moe_ffn_block(
      %input: tensor<?x?xf32>,          // [n_tokens, n_embd] (flattened batch*seq)
      %gate_inp_w: tensor<?x?xf32>,     // Router weights [n_expert, n_embd]
      %up_exps_w: tensor<?x?x?xf32>,    // Expert up [n_expert, n_ff, n_embd]
      %gate_exps_w: tensor<?x?x?xf32>,  // Expert gate [n_expert, n_ff, n_embd]
      %down_exps_w: tensor<?x?x?xf32>,  // Expert down [n_expert, n_embd, n_ff]
      %n_expert: index,                  // Total experts (8 for Mixtral)
      %n_expert_used: index,             // Top-k (2 for Mixtral)
      %n_embd: index,
      %n_ff: index,
      %normalize_weights: i1             // Whether to normalize (false for Mixtral)
  ) -> tensor<?x?xf32> {                 // [n_tokens, n_embd]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n_tokens = tensor.dim %input, %c0 : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32

    // Step 1: Router logits: [n_expert, n_tokens].
    // gate_inp_w is [n_expert, n_embd], input.T is [n_embd, n_tokens].
    %input_t_init = tensor.empty(%n_embd, %n_tokens) : tensor<?x?xf32>
    %input_t = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<?x?xf32>) outs(%input_t_init : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>

    %logits_init = tensor.empty(%n_expert, %n_tokens) : tensor<?x?xf32>
    %logits_filled = linalg.fill ins(%zero : f32) outs(%logits_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %logits = linalg.matmul ins(%gate_inp_w, %input_t : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%logits_filled : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Step 2: Softmax gating function along expert dimension (dim 0).
    %probs_init = tensor.empty(%n_expert, %n_tokens) : tensor<?x?xf32>
    %probs = linalg.softmax dimension(0) ins(%logits : tensor<?x?xf32>)
        outs(%probs_init : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Step 3: Top-k expert selection: returns weights and indices.
    // %weights: [n_expert_used, n_tokens] (top-k probabilities)
    // %selected_experts: [n_expert_used, n_tokens] (i32 indices)
    %weights_init = tensor.empty(%n_expert_used, %n_tokens) : tensor<?x?xf32>
    %indices_init = tensor.empty(%n_expert_used, %n_tokens) : tensor<?x?xi32>
    %weights, %selected_experts = iree_linalg_ext.topk
        dimension(0)
        ins(%probs : tensor<?x?xf32>)
        outs(%weights_init, %indices_init : tensor<?x?xf32>, tensor<?x?xi32>) {
      ^bb0(%lhs: f32, %rhs: f32):
        %cmp = arith.cmpf ogt, %lhs, %rhs : f32  // Descending order.
        iree_linalg_ext.yield %cmp : i1
    } -> tensor<?x?xf32>, tensor<?x?xi32>

    // Step 4: Conditional weight normalization (sum normalization per token).
    %weights_normalized = scf.if %normalize_weights -> (tensor<?x?xf32>) {
      // Sum weights across experts (dim 0) for each token.
      %sum_init = tensor.empty(%n_tokens) : tensor<?xf32>
      %sum_filled = linalg.fill ins(%zero : f32) outs(%sum_init : tensor<?xf32>) -> tensor<?xf32>

      %weights_sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>
        ],
        iterator_types = ["reduction", "parallel"]
      } ins(%weights : tensor<?x?xf32>) outs(%sum_filled : tensor<?xf32>) {
      ^bb0(%w: f32, %acc: f32):
        %sum = arith.addf %w, %acc : f32
        linalg.yield %sum : f32
      } -> tensor<?xf32>

      // Normalize by dividing each weight by sum.
      %weights_norm_init = tensor.empty(%n_expert_used, %n_tokens) : tensor<?x?xf32>
      %weights_norm = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%weights, %weights_sum : tensor<?x?xf32>, tensor<?xf32>)
        outs(%weights_norm_init : tensor<?x?xf32>) {
      ^bb0(%w: f32, %s: f32, %out: f32):
        %normalized = arith.divf %w, %s : f32
        linalg.yield %normalized : f32
      } -> tensor<?x?xf32>

      scf.yield %weights_norm : tensor<?x?xf32>
    } else {
      scf.yield %weights : tensor<?x?xf32>
    }

    // Step 5: Fused UP + GATE projection with SwiGLU.
    // input_t [n_embd, n_tokens], returns [n_ff, n_expert_used, n_tokens].
    %activated = util.call @moe_components.concat_gemm_id_silu(
        %input_t, %up_exps_w, %gate_exps_w, %selected_experts)
        : (tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xi32>)
        -> tensor<?x?x?xf32>

    // Step 6: Expert DOWN projection (gather + matmul).
    // Gather: down_exps_w[selected_experts] -> [n_expert_used, n_tokens, n_embd, n_ff]
    %gathered_down_init = tensor.empty(%n_expert_used, %n_tokens, %n_embd, %n_ff) : tensor<?x?x?x?xf32>
    %gathered_down = iree_linalg_ext.gather dimension_map = [0]
        ins(%down_exps_w, %selected_experts : tensor<?x?x?xf32>, tensor<?x?xi32>)
        outs(%gathered_down_init : tensor<?x?x?x?xf32>)
        -> tensor<?x?x?x?xf32>

    // Matmul: einsum("etoi,iet->oet", gathered_down, activated)
    // gathered_down[e, t, o, i] * activated[i, e, t] -> experts_out[o, e, t]
    %experts_out_init = tensor.empty(%n_embd, %n_expert_used, %n_tokens) : tensor<?x?x?xf32>
    %experts_out_filled = linalg.fill ins(%zero : f32) outs(%experts_out_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %experts_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>,  // gathered_down[e, t, o, i]
        affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>,       // activated[i, e, t]
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>        // experts_out[o, e, t]
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%gathered_down, %activated : tensor<?x?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%experts_out_filled : tensor<?x?x?xf32>) {
    ^bb0(%w: f32, %in: f32, %acc: f32):
      %prod = arith.mulf %w, %in : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?xf32>

    // Step 7: Apply expert weights element-wise.
    // Broadcast weights from [n_expert_used, n_tokens] to [n_embd, n_expert_used, n_tokens].
    %experts_weighted_init = tensor.empty(%n_embd, %n_expert_used, %n_tokens) : tensor<?x?x?xf32>
    %experts_weighted = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%experts_out, %weights_normalized : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%experts_weighted_init : tensor<?x?x?xf32>) {
    ^bb0(%expert_val: f32, %weight: f32, %out: f32):
      %weighted = arith.mulf %expert_val, %weight : f32
      linalg.yield %weighted : f32
    } -> tensor<?x?x?xf32>

    // Step 8: Sum expert outputs across expert dimension (reduce dim 1).
    %output_init = tensor.empty(%n_embd, %n_tokens) : tensor<?x?xf32>
    %output_filled = linalg.fill ins(%zero : f32) outs(%output_init : tensor<?x?xf32>) -> tensor<?x?xf32>

    %output_summed = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%experts_weighted : tensor<?x?x?xf32>) outs(%output_filled : tensor<?x?xf32>) {
    ^bb0(%expert_val: f32, %acc: f32):
      %sum = arith.addf %expert_val, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>

    // Transpose back to [n_tokens, n_embd].
    %final_init = tensor.empty(%n_tokens, %n_embd) : tensor<?x?xf32>
    %final = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%output_summed : tensor<?x?xf32>) outs(%final_init : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>

    util.return %final : tensor<?x?xf32>
  }

}
