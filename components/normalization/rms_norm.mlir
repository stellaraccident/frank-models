// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reusable RMS Normalization implementations.
// RMS norm: out = (x / sqrt(mean(x^2) + eps)) * weight
//
// Usage:
//   %output = call @rms_norm_linalg(%input, %weight, %eps)
//       : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

module @rms_norm_components {

  // Standard RMS norm with linalg operations.
  // Input: [batch, hidden_dim], Weight: [hidden_dim], Output: [batch, hidden_dim]
  util.func public @rms_norm_linalg(
      %input: tensor<?x?xf32>,
      %weight: tensor<?xf32>,
      %eps: f32
  ) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>
    %dim1 = tensor.dim %input, %c1 : tensor<?x?xf32>

    // Step 1: Compute sum of squares for each row.
    %init_sum = tensor.empty(%dim0) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero : f32) outs(%init_sum : tensor<?xf32>) -> tensor<?xf32>

    %sum_sq = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : tensor<?x?xf32>) outs(%sum_init : tensor<?xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sq = arith.mulf %in, %in : f32
      %sum = arith.addf %acc, %sq : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>

    // Step 2: Compute RMS = sqrt(mean(x^2) + eps).
    %dim1_i32 = arith.index_cast %dim1 : index to i32
    %dim1_f32 = arith.sitofp %dim1_i32 : i32 to f32

    %rms_init = tensor.empty(%dim0) : tensor<?xf32>
    %rms = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%sum_sq : tensor<?xf32>) outs(%rms_init : tensor<?xf32>) {
    ^bb0(%sum: f32, %out: f32):
      %mean = arith.divf %sum, %dim1_f32 : f32
      %with_eps = arith.addf %mean, %eps : f32
      %rms_val = math.sqrt %with_eps : f32
      linalg.yield %rms_val : f32
    } -> tensor<?xf32>

    // Step 3: Normalize and scale by weight.
    %output_init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %rms, %weight : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
      outs(%output_init : tensor<?x?xf32>) {
    ^bb0(%x: f32, %rms_val: f32, %w: f32, %out: f32):
      %normalized = arith.divf %x, %rms_val : f32
      %scaled = arith.mulf %normalized, %w : f32
      linalg.yield %scaled : f32
    } -> tensor<?x?xf32>

    util.return %output : tensor<?x?xf32>
  }

  // RMS norm with explicit fusion boundary.
  // Use this variant when you need to control kernel boundaries.
  util.func public @rms_norm_fused(
      %input: tensor<?x?xf32>,
      %weight: tensor<?xf32>,
      %eps: f32
  ) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>
    %dim1 = tensor.dim %input, %c1 : tensor<?x?xf32>

    %output = flow.dispatch.region -> (tensor<?x?xf32>{%dim0, %dim1}) {
      %init_sum = tensor.empty(%dim0) : tensor<?xf32>
      %zero = arith.constant 0.0 : f32
      %sum_init = linalg.fill ins(%zero : f32) outs(%init_sum : tensor<?xf32>) -> tensor<?xf32>

      %sum_sq = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<?x?xf32>) outs(%sum_init : tensor<?xf32>) {
      ^bb0(%in: f32, %acc: f32):
        %sq = arith.mulf %in, %in : f32
        %sum = arith.addf %acc, %sq : f32
        linalg.yield %sum : f32
      } -> tensor<?xf32>

      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %dim1_f32 = arith.sitofp %dim1_i32 : i32 to f32

      %rms_init = tensor.empty(%dim0) : tensor<?xf32>
      %rms = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%sum_sq : tensor<?xf32>) outs(%rms_init : tensor<?xf32>) {
      ^bb0(%sum: f32, %out: f32):
        %mean = arith.divf %sum, %dim1_f32 : f32
        %with_eps = arith.addf %mean, %eps : f32
        %rms_val = math.sqrt %with_eps : f32
        linalg.yield %rms_val : f32
      } -> tensor<?xf32>

      %output_init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%input, %rms, %weight : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
        outs(%output_init : tensor<?x?xf32>) {
      ^bb0(%x: f32, %rms_val: f32, %w: f32, %out: f32):
        %normalized = arith.divf %x, %rms_val : f32
        %scaled = arith.mulf %normalized, %w : f32
        linalg.yield %scaled : f32
      } -> tensor<?x?xf32>

      flow.return %result : tensor<?x?xf32>
    }

    util.return %output : tensor<?x?xf32>
  }

}
