// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Indirect matrix multiply for expert selection (ggml_mul_mat_id decomposition).
// Reference: llama.cpp ggml_mul_mat_id.
//
// Dynamically selects expert weight matrices based on indices and performs
// batched matrix multiplication. For MoE layers: each token uses different
// expert weights based on routing decisions.
//
// For each token t and each selected expert slot e:
//   expert_id = ids[e, t]
//   result[:, e, t] = weights[expert_id] @ input[:, e, t]
//
// Decomposed into: gather → reshape → batch_matmul → reshape.

module @moe_components {

  util.func public @mul_mat_id(
      %weights: tensor<?x?x?xf32>,      // Expert weights [n_expert, n_out, n_in]
      %input: tensor<?x?x?xf32>,        // Input [n_in, n_expert_used, n_tokens]
      %ids: tensor<?x?xi32>             // Expert indices [n_expert_used, n_tokens]
  ) -> tensor<?x?x?xf32> {              // Output [n_out, n_expert_used, n_tokens]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %n_expert = tensor.dim %weights, %c0 : tensor<?x?x?xf32>
    %n_out = tensor.dim %weights, %c1 : tensor<?x?x?xf32>
    %n_in = tensor.dim %weights, %c2 : tensor<?x?x?xf32>
    %n_expert_used = tensor.dim %input, %c1 : tensor<?x?x?xf32>
    %n_tokens = tensor.dim %input, %c2 : tensor<?x?x?xf32>

    // Step 1: Flatten indices for batched gather: [n_expert_used * n_tokens].
    %batch_size = arith.muli %n_expert_used, %n_tokens : index
    %ids_flat = tensor.collapse_shape %ids [[0, 1]]
      : tensor<?x?xi32> into tensor<?xi32>

    // Step 2: Gather expert matrices based on flattened indices.
    // Weights are [n_expert, n_out, n_in], gather along dim 0.
    // Output: [n_expert_used * n_tokens, n_out, n_in].
    %gathered_init = tensor.empty(%batch_size, %n_out, %n_in) : tensor<?x?x?xf32>
    %weights_gathered = iree_linalg_ext.gather dimension_map = [0]
      ins(%weights, %ids_flat : tensor<?x?x?xf32>, tensor<?xi32>)
      outs(%gathered_init : tensor<?x?x?xf32>)
      -> tensor<?x?x?xf32>

    // Step 3: Reshape input for batched matmul.
    // Transpose input: [n_in, n_expert_used, n_tokens] -> [n_expert_used, n_tokens, n_in].
    %input_perm_init = tensor.empty(%n_expert_used, %n_tokens, %n_in) : tensor<?x?x?xf32>
    %input_perm = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2, d0)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<?x?x?xf32>) outs(%input_perm_init : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>

    // Flatten to [n_expert_used * n_tokens, n_in].
    %input_flat = tensor.collapse_shape %input_perm [[0, 1], [2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>

    // Expand to [n_expert_used * n_tokens, n_in, 1] for batched matmul.
    %input_batched = tensor.expand_shape %input_flat [[0], [1, 2]]
      output_shape [%batch_size, %n_in, %c1]
      : tensor<?x?xf32> into tensor<?x?x1xf32>

    // Step 4: Batched matrix-vector multiply.
    // [batch, n_out, n_in] @ [batch, n_in, 1] -> [batch, n_out, 1]
    %zero = arith.constant 0.0 : f32
    %result_batched_init = tensor.empty(%batch_size, %n_out) : tensor<?x?x1xf32>
    %result_batched_filled = linalg.fill ins(%zero : f32) outs(%result_batched_init : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %result_batched = linalg.batch_matmul
      ins(%weights_gathered, %input_batched : tensor<?x?x?xf32>, tensor<?x?x1xf32>)
      outs(%result_batched_filled : tensor<?x?x1xf32>)
      -> tensor<?x?x1xf32>

    // Step 5: Reshape to final output [n_out, n_expert_used, n_tokens].
    // Collapse [batch, n_out, 1] to [batch, n_out].
    %result_squeezed = tensor.collapse_shape %result_batched [[0], [1, 2]]
      : tensor<?x?x1xf32> into tensor<?x?xf32>

    // Expand [batch, n_out] to [n_expert_used, n_tokens, n_out].
    %result_expanded = tensor.expand_shape %result_squeezed [[0, 1], [2]]
      output_shape [%n_expert_used, %n_tokens, %n_out]
      : tensor<?x?xf32> into tensor<?x?x?xf32>

    // Transpose to [n_out, n_expert_used, n_tokens].
    %final_init = tensor.empty(%n_out, %n_expert_used, %n_tokens) : tensor<?x?x?xf32>
    %final = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%result_expanded : tensor<?x?x?xf32>) outs(%final_init : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>

    util.return %final : tensor<?x?x?xf32>
  }

}
