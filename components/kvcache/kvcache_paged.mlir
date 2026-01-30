// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Paged KV Cache - flat block pool with indirect addressing.
//
// Physical cache layout (flat pool, shared across all sequences):
//   dim 0: block_idx    - Physical block index in the pool
//   dim 1: pos_in_block - Position within block [0, block_size)
//   dim 2: n_head_kv    - Number of KV heads
//   dim 3: head_dim     - Dimension per attention head
//
// Interface asymmetry (intentional):
//   - gather: reads full context, flat [n_ctx_tokens] block indices
//   - scatter_single_token: writes ONE token per sequence (decode hot path)
//
// Cache object structure (internal):
//   this[0] = K cache as !hal.buffer_view [n_blocks, block_size, n_head_kv, head_dim]
//   this[1] = V cache as !hal.buffer_view [n_blocks, block_size, n_head_kv, head_dim]
//
// Usage:
//   %cache = call @allocate(%n_blocks, %block_size, %n_head_kv, %head_dim)
//   %k, %v = call @gather(%cache, %block_indices, %block_size)
//   %updated = call @scatter_single_token(%cache, %new_k, %new_v, %block_indices, %pos_in_blocks)

// Cache element type - change here to specialize (e.g., f16, bf16)
!elem_t = f32

module @kvcache_paged {

  // Allocate a new paged KV cache (block pool).
  //
  // Args:
  //   %n_blocks:   Number of physical blocks in the pool
  //   %block_size: Tokens per block
  //   %n_head_kv:  Number of KV attention heads
  //   %head_dim:   Dimension per attention head
  //
  // Returns:
  //   !util.list<?> containing K and V block pools,
  //   each [n_blocks, block_size, n_head_kv, head_dim]
  util.func public @allocate(
      %n_blocks: index,
      %block_size: index,
      %n_head_kv: index,
      %head_dim: index
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %element_size = util.sizeof !elem_t
    %affinity = arith.constant -1 : i64

    // Compute buffer size: n_blocks * block_size * n_head_kv * head_dim * sizeof(element)
    %d0 = arith.muli %n_blocks, %block_size : index
    %d1 = arith.muli %d0, %n_head_kv : index
    %d2 = arith.muli %d1, %head_dim : index
    %byte_size = arith.muli %d2, %element_size : index

    // Get device and allocator
    %device = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator

    // Allocate K block pool
    %memory_type = hal.memory_type<"DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|DispatchStorageRead|DispatchStorageWrite"> : i32
    %k_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%byte_size}

    // Create K buffer view [n_blocks, block_size, n_head_kv, head_dim]
    %element_type = hal.element_type<!elem_t> : i32
    %encoding_type = hal.encoding_type<dense_row_major> : i32
    %k_bv = hal.buffer_view.create buffer(%k_buffer : !hal.buffer)[%c0, %byte_size]
                                   shape([%n_blocks, %block_size, %n_head_kv, %head_dim])
                                   type(%element_type)
                                   encoding(%encoding_type) : !hal.buffer_view

    // Allocate V block pool (distinct allocation)
    %v_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%byte_size}

    // Create V buffer view
    %v_bv = hal.buffer_view.create buffer(%v_buffer : !hal.buffer)[%c0, %byte_size]
                                   shape([%n_blocks, %block_size, %n_head_kv, %head_dim])
                                   type(%element_type)
                                   encoding(%encoding_type) : !hal.buffer_view

    // Create list and store buffer views
    %this = util.list.create %c2 : !util.list<?>
    util.list.resize %this, %c2 : !util.list<?>
    util.list.set %this[%c0], %k_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %this[%c1], %v_bv : !hal.buffer_view -> !util.list<?>

    util.return %this : !util.list<?>
  }

  // Gather K and V from paged cache using flat block indices.
  //
  // Uses linalg.generic + tensor.extract pattern that auto-vectorizes
  // to iree_vector_ext.transfer_gather on GPU.
  //
  // Args:
  //   %this:          Cache object from @allocate
  //   %block_indices: [n_ctx_tokens] i32 - physical block index per context token (flat)
  //   %block_size:    Tokens per block (for computing position within block)
  //
  // Returns:
  //   %k: [n_ctx_tokens, n_head_kv, head_dim]
  //   %v: [n_ctx_tokens, n_head_kv, head_dim]
  util.func public @gather(
      %this: !util.list<?>,
      %block_indices: tensor<?xi32>,
      %block_size: index
  ) -> (tensor<?x?x?x!elem_t>, tensor<?x?x?x!elem_t>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Get n_ctx_tokens from block_indices [n_ctx_tokens]
    %n_ctx_tokens = tensor.dim %block_indices, %c0 : tensor<?xi32>

    // Get K buffer view and extract dimensions
    %k_bv = util.list.get %this[%c0] : !util.list<?> -> !hal.buffer_view
    %n_blocks = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %block_size_from_cache = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %n_head_kv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %head_dim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[3] : index

    // Import K cache [n_blocks, block_size, n_head_kv, head_dim]
    %k_cache = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size_from_cache, %n_head_kv, %head_dim}

    // Allocate output tensor [n_ctx_tokens, n_head_kv, head_dim]
    %k_init = tensor.empty(%n_ctx_tokens, %n_head_kv, %head_dim) : tensor<?x?x?x!elem_t>

    // Gather K using linalg.generic + tensor.extract pattern
    // This auto-vectorizes to iree_vector_ext.transfer_gather on GPU
    %k_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(ctx, head, dim) -> (ctx)>,            // block_indices
        affine_map<(ctx, head, dim) -> (ctx, head, dim)>  // output
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%block_indices : tensor<?xi32>)
      outs(%k_init : tensor<?x?x?x!elem_t>) {
    ^bb0(%block_idx_i32: i32, %out: !elem_t):
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %ctx_idx = linalg.index 0 : index
      %head_idx = linalg.index 1 : index
      %dim_idx = linalg.index 2 : index

      // Position within block = ctx_idx % block_size
      %pos_in_block = arith.remui %ctx_idx, %block_size : index

      // Gather from cache[block_idx, pos_in_block, head_idx, dim_idx]
      %k_val = tensor.extract %k_cache[%block_idx, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>

      linalg.yield %k_val : !elem_t
    } -> tensor<?x?x?x!elem_t>

    // Get V buffer view and import
    %v_bv = util.list.get %this[%c1] : !util.list<?> -> !hal.buffer_view
    %v_cache = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size_from_cache, %n_head_kv, %head_dim}

    // Allocate V output
    %v_init = tensor.empty(%n_ctx_tokens, %n_head_kv, %head_dim) : tensor<?x?x?x!elem_t>

    // Gather V
    %v_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(ctx, head, dim) -> (ctx)>,
        affine_map<(ctx, head, dim) -> (ctx, head, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%block_indices : tensor<?xi32>)
      outs(%v_init : tensor<?x?x?x!elem_t>) {
    ^bb0(%block_idx_i32: i32, %out: !elem_t):
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %ctx_idx = linalg.index 0 : index
      %head_idx = linalg.index 1 : index
      %dim_idx = linalg.index 2 : index
      %pos_in_block = arith.remui %ctx_idx, %block_size : index
      %v_val = tensor.extract %v_cache[%block_idx, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>
      linalg.yield %v_val : !elem_t
    } -> tensor<?x?x?x!elem_t>

    util.return %k_gathered, %v_gathered : tensor<?x?x?x!elem_t>, tensor<?x?x?x!elem_t>
  }

  // Scatter one new token per sequence to paged cache.
  //
  // This is the decode-phase operation where each sequence in the batch
  // adds exactly one token. For multi-token scatter (prefill), see future
  // @scatter operation.
  //
  // Args:
  //   %this:          Cache object from @allocate
  //   %new_k:         [batch, n_head_kv, head_dim] - ONE new K token per sequence
  //   %new_v:         [batch, n_head_kv, head_dim] - ONE new V token per sequence
  //   %block_indices: [batch] i32 - which block for each sequence
  //   %pos_in_blocks: [batch] i32 - position within block for each sequence
  //
  // Returns:
  //   Updated cache object
  util.func public @scatter_single_token(
      %this: !util.list<?>,
      %new_k: tensor<?x?x?x!elem_t>,
      %new_v: tensor<?x?x?x!elem_t>,
      %block_indices: tensor<?xi32>,
      %pos_in_blocks: tensor<?xi32>
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Get dimensions from new_k [batch, n_head_kv, head_dim]
    %batch_size = tensor.dim %new_k, %c0 : tensor<?x?x?x!elem_t>
    %n_head_kv = tensor.dim %new_k, %c1 : tensor<?x?x?x!elem_t>
    %head_dim = tensor.dim %new_k, %c2 : tensor<?x?x?x!elem_t>

    // Get K buffer view and import
    %k_bv = util.list.get %this[%c0] : !util.list<?> -> !hal.buffer_view
    %n_blocks = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %block_size = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %k_cache = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Scatter K: one token per sequence
    %k_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%cache_k = %k_cache) -> (tensor<?x?x?x?x!elem_t>) {

      // Get block index and position for this sequence
      %block_idx_i32 = tensor.extract %block_indices[%i] : tensor<?xi32>
      %pos_i32 = tensor.extract %pos_in_blocks[%i] : tensor<?xi32>
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %pos = arith.index_cast %pos_i32 : i32 to index

      // Extract this sequence's K values [1, n_head_kv, head_dim]
      %new_k_slice = tensor.extract_slice %new_k[%i, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?x!elem_t> to tensor<1x?x?x!elem_t>

      // Insert into cache at [block_idx, pos, :, :]
      %updated = tensor.insert_slice %new_k_slice into %cache_k[%block_idx, %pos, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %updated : tensor<?x?x?x?x!elem_t>
    }

    // Tie dimensions after loop (IREE needs explicit shape info)
    %k_updated_tied = flow.tensor.reshape %k_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Export updated K
    %k_updated_bv = hal.tensor.export %k_updated_tied : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Get V buffer view and import
    %v_bv = util.list.get %this[%c1] : !util.list<?> -> !hal.buffer_view
    %v_cache = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Scatter V: one token per sequence
    %v_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%cache_v = %v_cache) -> (tensor<?x?x?x?x!elem_t>) {

      %block_idx_i32 = tensor.extract %block_indices[%i] : tensor<?xi32>
      %pos_i32 = tensor.extract %pos_in_blocks[%i] : tensor<?xi32>
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %pos = arith.index_cast %pos_i32 : i32 to index

      %new_v_slice = tensor.extract_slice %new_v[%i, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?x!elem_t> to tensor<1x?x?x!elem_t>

      %updated = tensor.insert_slice %new_v_slice into %cache_v[%block_idx, %pos, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %updated : tensor<?x?x?x?x!elem_t>
    }

    // Tie dimensions after loop
    %v_updated_tied = flow.tensor.reshape %v_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Export updated V
    %v_updated_bv = hal.tensor.export %v_updated_tied : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Update list with new buffer views
    util.list.set %this[%c0], %k_updated_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %this[%c1], %v_updated_bv : !hal.buffer_view -> !util.list<?>

    util.return %this : !util.list<?>
  }

}
