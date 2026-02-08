// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Embedding lookup: gather embedding vectors by token indices.
//
// Usage:
//   %output = call @embedding_lookup(%weight, %indices)
//       : (tensor<?x?xf32>, tensor<?x?xi64>) -> tensor<?x?x?xf32>

module @embedding_components {

  util.func public @embedding_lookup(
      %weight: tensor<?x?xf32>,   // [vocab_size, n_embd]
      %indices: tensor<?x?xi64>   // [batch, seq_len]
  ) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %batch = tensor.dim %indices, %c0 : tensor<?x?xi64>
    %seq_len = tensor.dim %indices, %c1 : tensor<?x?xi64>
    %n_embd = tensor.dim %weight, %c1 : tensor<?x?xf32>

    // Flatten indices from [batch, seq_len] to [batch * seq_len].
    %batch_seq = arith.muli %batch, %seq_len : index
    %indices_flat = tensor.collapse_shape %indices [[0, 1]] : tensor<?x?xi64> into tensor<?xi64>

    // Gather embeddings: [batch * seq_len, n_embd].
    %init_flat = tensor.empty(%batch_seq, %n_embd) : tensor<?x?xf32>
    %embeddings_flat = iree_linalg_ext.gather dimension_map = [0]
        ins(%weight, %indices_flat : tensor<?x?xf32>, tensor<?xi64>)
        outs(%init_flat : tensor<?x?xf32>) -> tensor<?x?xf32>

    // Unflatten back to [batch, seq_len, n_embd].
    %embeddings = tensor.expand_shape %embeddings_flat [[0, 1], [2]] output_shape [%batch, %seq_len, %n_embd]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    util.return %embeddings : tensor<?x?x?xf32>
  }

  // 1D variant: single token per batch (for decode step).
  // Takes indices of shape [batch] and returns embeddings of shape [batch, n_embd].
  util.func public @embedding_lookup_1d(
      %weight: tensor<?x?xf32>,   // [vocab_size, n_embd]
      %indices: tensor<?xi64>     // [batch]
  ) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %batch = tensor.dim %indices, %c0 : tensor<?xi64>
    %n_embd = tensor.dim %weight, %c1 : tensor<?x?xf32>

    // Gather embeddings directly: [batch, n_embd].
    %init = tensor.empty(%batch, %n_embd) : tensor<?x?xf32>
    %embeddings = iree_linalg_ext.gather dimension_map = [0]
        ins(%weight, %indices : tensor<?x?xf32>, tensor<?xi64>)
        outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>

    util.return %embeddings : tensor<?x?xf32>
  }

}
