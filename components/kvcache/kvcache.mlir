// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical Paged KV Cache with Device-Side Control Data
//
// This is the canonical implementation following Ben's device-first philosophy:
// all control metadata (block_tables, context_lens, block_indices, pos_in_blocks)
// stays on device as tensor<>, not on host as !util.list<>.
//
// The linalg.generic + tensor.extract pattern auto-vectorizes to
// iree_vector_ext.transfer_gather on GPU. Current codegen may produce
// inefficient code with tensor.extract, but the interface is correct -
// codegen will be fixed later.
//
// Physical cache layout (unified block pool, shared across all layers):
//   K/V: [n_blocks, block_size, n_head_kv, head_dim]
//
// Layer-aware metadata:
//   block_tables: [n_layers, batch, max_blocks_per_seq] - physical block mapping
//   context_lens: [n_layers, batch] - context length per layer/sequence
//
// Cache object structure:
//   cache[0] = K blocks as !hal.buffer_view
//   cache[1] = V blocks as !hal.buffer_view
//
// Interface:
//   allocate(n_blocks, block_size, n_head_kv, head_dim) -> cache
//   gather(cache, layer, block_tables, context_lens, max_context_len) -> (K, V)
//   scatter_decode(cache, layer, new_k, new_v, block_indices, pos_in_blocks) -> cache
//   scatter_prefill(cache, layer, new_k, new_v, block_tables, start_positions, block_size) -> cache

!elem_t = f32

module @kvcache {

  // Allocate a paged KV cache block pool.
  //
  // Args:
  //   %n_blocks:   Total physical blocks in the pool
  //   %block_size: Tokens per block
  //   %n_head_kv:  Number of KV attention heads
  //   %head_dim:   Dimension per attention head
  //
  // Returns:
  //   !util.list<?> containing K and V block pools
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

  // Gather K and V for attention using device-side block tables and context lengths.
  //
  // All control data stays on device. The gather uses linalg.generic + tensor.extract
  // which auto-vectorizes to iree_vector_ext.transfer_gather on GPU.
  //
  // Logical shapes:
  //   cache:           !util.list<?> containing K_blocks, V_blocks
  //     K_blocks:      [n_blocks, block_size, n_head_kv, head_dim] - physical storage
  //     V_blocks:      [n_blocks, block_size, n_head_kv, head_dim]
  //   layer:           index - which transformer layer (0 to n_layers-1)
  //   block_tables:    [n_layers, batch, max_blocks_per_seq] i32 - block indirection
  //   context_lens:    [n_layers, batch] i32 - actual context length per layer/sequence
  //   max_context_len: index - output sequence dimension
  //
  // Returns:
  //   k_gathered:      [batch, max_context_len, n_head_kv, head_dim]
  //   v_gathered:      [batch, max_context_len, n_head_kv, head_dim]
  //   Positions >= context_lens[layer, b] are zeroed.
  util.func public @gather(
      %cache: !util.list<?>,
      %layer: index,
      %block_tables: tensor<?x?x?xi32>,
      %context_lens: tensor<?x?xi32>,
      %max_context_len: index
  ) -> (tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : !elem_t

    // Get batch size and max_blocks from block_tables [n_layers, batch, max_blocks]
    %batch = tensor.dim %block_tables, %c1 : tensor<?x?x?xi32>
    %max_blocks = tensor.dim %block_tables, %c2 : tensor<?x?x?xi32>

    // Slice block_tables and context_lens for this layer
    // block_tables_layer: [batch, max_blocks]
    %block_tables_layer = tensor.extract_slice %block_tables[%layer, 0, 0] [1, %batch, %max_blocks] [1, 1, 1]
      : tensor<?x?x?xi32> to tensor<1x?x?xi32>
    %block_tables_2d = tensor.collapse_shape %block_tables_layer [[0, 1], [2]]
      : tensor<1x?x?xi32> into tensor<?x?xi32>

    // context_lens_layer: [batch]
    %context_lens_layer = tensor.extract_slice %context_lens[%layer, 0] [1, %batch] [1, 1]
      : tensor<?x?xi32> to tensor<1x?xi32>
    %context_lens_1d = tensor.collapse_shape %context_lens_layer [[0, 1]]
      : tensor<1x?xi32> into tensor<?xi32>

    // Import K cache
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %n_blocks = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %block_size = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %n_head_kv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %head_dim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[3] : index
    %k_blocks = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Import V cache
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %v_blocks = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Allocate output tensors [batch, max_context_len, n_head_kv, head_dim]
    %k_init = tensor.empty(%batch, %max_context_len, %n_head_kv, %head_dim) : tensor<?x?x?x?x!elem_t>
    %v_init = tensor.empty(%batch, %max_context_len, %n_head_kv, %head_dim) : tensor<?x?x?x?x!elem_t>

    // Gather K using linalg.generic + tensor.extract pattern
    // This auto-vectorizes to iree_vector_ext.transfer_gather on GPU
    %k_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(b, ctx, head, dim) -> (b)>,                // context_lens
        affine_map<(b, ctx, head, dim) -> (b, ctx, head, dim)> // output
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%context_lens_1d : tensor<?xi32>)
      outs(%k_init : tensor<?x?x?x?x!elem_t>) {
    ^bb0(%ctx_len_i32: i32, %out: !elem_t):
      %b_idx = linalg.index 0 : index
      %ctx_idx = linalg.index 1 : index
      %head_idx = linalg.index 2 : index
      %dim_idx = linalg.index 3 : index

      // Compute logical block index and position within block
      %logical_block = arith.divui %ctx_idx, %block_size : index
      %pos_in_block = arith.remui %ctx_idx, %block_size : index

      // Look up physical block from block_tables_2d[b_idx, logical_block]
      // NOTE: tensor.extract on device tensor - this is intentional
      %physical_block_i32 = tensor.extract %block_tables_2d[%b_idx, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      // Gather from K cache
      %k_val = tensor.extract %k_blocks[%physical_block, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>

      // Mask out positions beyond context_len
      %ctx_len = arith.index_cast %ctx_len_i32 : i32 to index
      %in_range = arith.cmpi ult, %ctx_idx, %ctx_len : index
      %result = arith.select %in_range, %k_val, %zero : !elem_t

      linalg.yield %result : !elem_t
    } -> tensor<?x?x?x?x!elem_t>

    // Gather V similarly
    %v_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(b, ctx, head, dim) -> (b)>,
        affine_map<(b, ctx, head, dim) -> (b, ctx, head, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%context_lens_1d : tensor<?xi32>)
      outs(%v_init : tensor<?x?x?x?x!elem_t>) {
    ^bb0(%ctx_len_i32: i32, %out: !elem_t):
      %b_idx = linalg.index 0 : index
      %ctx_idx = linalg.index 1 : index
      %head_idx = linalg.index 2 : index
      %dim_idx = linalg.index 3 : index

      %logical_block = arith.divui %ctx_idx, %block_size : index
      %pos_in_block = arith.remui %ctx_idx, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%b_idx, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %v_val = tensor.extract %v_blocks[%physical_block, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>

      %ctx_len = arith.index_cast %ctx_len_i32 : i32 to index
      %in_range = arith.cmpi ult, %ctx_idx, %ctx_len : index
      %result = arith.select %in_range, %v_val, %zero : !elem_t

      linalg.yield %result : !elem_t
    } -> tensor<?x?x?x?x!elem_t>

    util.return %k_gathered, %v_gathered : tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>
  }

  // Scatter one new token per sequence to paged cache (decode phase).
  //
  // All control data stays on device as tensors. The scatter loop uses
  // tensor.extract to read indices - this is intentional for interface
  // correctness; codegen will be optimized later.
  //
  // Logical shapes:
  //   cache:           !util.list<?> containing K_blocks, V_blocks
  //     K_blocks:      [n_blocks, block_size, n_head_kv, head_dim] - physical storage
  //     V_blocks:      [n_blocks, block_size, n_head_kv, head_dim]
  //   layer:           index - which transformer layer (0 to n_layers-1)
  //   new_k:           [batch, n_head_kv, head_dim] - one K vector per sequence
  //   new_v:           [batch, n_head_kv, head_dim] - one V vector per sequence
  //   block_indices:   [batch] i32 - which physical block for each sequence
  //   pos_in_blocks:   [batch] i32 - position within block (0 to block_size-1)
  //
  // Returns:
  //   Updated cache object
  //
  // Note: layer parameter included for API consistency with gather, though
  // scatter_decode writes directly to physical blocks (no layer slicing needed).
  util.func public @scatter_decode(
      %cache: !util.list<?>,
      %layer: index,
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

    // Import K cache
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %n_blocks = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %block_size = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %k_blocks = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Scatter K: loop over batch
    // NOTE: tensor.extract on device tensors is intentional - interface correctness first
    %k_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%cache_k = %k_blocks) -> (tensor<?x?x?x?x!elem_t>) {

      // Get block index and position for this sequence (device tensor reads)
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

    // Tie dimensions after loop
    %k_updated_tied = flow.tensor.reshape %k_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Export updated K
    %k_updated_bv = hal.tensor.export %k_updated_tied : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Import and scatter V
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %v_blocks = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    %v_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%cache_v = %v_blocks) -> (tensor<?x?x?x?x!elem_t>) {

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
    util.list.set %cache[%c0], %k_updated_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %v_updated_bv : !hal.buffer_view -> !util.list<?>

    util.return %cache : !util.list<?>
  }

  // Scatter multiple tokens K/V to cache (prefill phase).
  //
  // Uses flatten approach: single loop over batch*seq_len index space.
  // This assumes uniform sequence lengths (all sequences have length = seq_len
  // dimension of new_k). For variable-length sequences, pad to uniform length
  // or process batches separately by length.
  //
  // NOTE: Ragged scatter (per-sequence seq_lengths) is a potential future
  // optimization. Current IREE shape tracking limitations prevent scf.if
  // with tensor returns inside loops.
  //
  // Logical shapes:
  //   cache:           !util.list<?> containing K_blocks, V_blocks
  //     K_blocks:      [n_blocks, block_size, n_head_kv, head_dim] - physical storage
  //     V_blocks:      [n_blocks, block_size, n_head_kv, head_dim]
  //   layer:           index - which transformer layer (for block_tables slicing)
  //   new_k:           [batch, seq_len, n_head_kv, head_dim] - K values to scatter
  //   new_v:           [batch, seq_len, n_head_kv, head_dim] - V values to scatter
  //   block_tables:    [n_layers, batch, max_blocks_per_seq] i32 - block indirection
  //   start_positions: [batch] i32 - starting position in sequence (usually 0)
  //   block_size:      index - tokens per block
  //
  // Returns:
  //   Updated cache object with K/V written
  //
  // Note: seq_lengths parameter removed - uses tensor dim directly.
  // For variable-length support, pad sequences or batch by length.
  util.func public @scatter_prefill(
      %cache: !util.list<?>,
      %layer: index,
      %new_k: tensor<?x?x?x?x!elem_t>,    // [batch, seq_len, n_head_kv, head_dim]
      %new_v: tensor<?x?x?x?x!elem_t>,
      %block_tables: tensor<?x?x?xi32>,    // [n_layers, batch, max_blocks]
      %start_positions: tensor<?xi32>,      // [batch]
      %block_size: index
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Get dimensions from new_k [batch, seq_len, n_head_kv, head_dim]
    %batch = tensor.dim %new_k, %c0 : tensor<?x?x?x?x!elem_t>
    %seq_len = tensor.dim %new_k, %c1 : tensor<?x?x?x?x!elem_t>
    %n_head_kv = tensor.dim %new_k, %c2 : tensor<?x?x?x?x!elem_t>
    %head_dim = tensor.dim %new_k, %c3 : tensor<?x?x?x?x!elem_t>

    // Slice block_tables for this layer: [batch, max_blocks]
    %max_blocks = tensor.dim %block_tables, %c2 : tensor<?x?x?xi32>
    %block_tables_layer = tensor.extract_slice %block_tables[%layer, 0, 0] [1, %batch, %max_blocks] [1, 1, 1]
      : tensor<?x?x?xi32> to tensor<1x?x?xi32>
    %block_tables_2d = tensor.collapse_shape %block_tables_layer [[0, 1], [2]]
      : tensor<1x?x?xi32> into tensor<?x?xi32>

    // Import K cache
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %n_blocks = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %cache_block_size = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %k_blocks = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim}

    // Total positions to process (flatten batch * seq_len)
    %total_positions = arith.muli %batch, %seq_len : index

    // Scatter K: single loop over flattened index space (no conditional)
    %k_updated = scf.for %flat_idx = %c0 to %total_positions step %c1
        iter_args(%cache_k = %k_blocks) -> (tensor<?x?x?x?x!elem_t>) {

      // Compute batch and seq indices from flat index
      %b = arith.divui %flat_idx, %seq_len : index
      %s = arith.remui %flat_idx, %seq_len : index

      // Compute absolute position and block indices
      %start_pos_i32 = tensor.extract %start_positions[%b] : tensor<?xi32>
      %start_pos = arith.index_cast %start_pos_i32 : i32 to index
      %abs_pos = arith.addi %start_pos, %s : index
      %logical_block = arith.divui %abs_pos, %block_size : index
      %pos_in_block = arith.remui %abs_pos, %block_size : index

      // Lookup physical block from block_tables
      %physical_block_i32 = tensor.extract %block_tables_2d[%b, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      // Extract K value for this position: [1, 1, n_head_kv, head_dim]
      %new_k_slice = tensor.extract_slice %new_k[%b, %s, 0, 0] [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> to tensor<1x1x?x?x!elem_t>

      // Collapse to [1, n_head_kv, head_dim] for insert
      %new_k_3d = tensor.collapse_shape %new_k_slice [[0, 1], [2], [3]]
        : tensor<1x1x?x?x!elem_t> into tensor<1x?x?x!elem_t>

      // Insert into cache at [physical_block, pos_in_block, :, :]
      %result = tensor.insert_slice %new_k_3d into %cache_k[%physical_block, %pos_in_block, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %result : tensor<?x?x?x?x!elem_t>
    }

    // Tie dimensions after loop
    %k_updated_tied = flow.tensor.reshape %k_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim} -> tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim}

    // Export updated K
    %k_updated_bv = hal.tensor.export %k_updated_tied : tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Import and scatter V
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %v_blocks = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim}

    %v_updated = scf.for %flat_idx = %c0 to %total_positions step %c1
        iter_args(%cache_v = %v_blocks) -> (tensor<?x?x?x?x!elem_t>) {

      %b = arith.divui %flat_idx, %seq_len : index
      %s = arith.remui %flat_idx, %seq_len : index

      %start_pos_i32 = tensor.extract %start_positions[%b] : tensor<?xi32>
      %start_pos = arith.index_cast %start_pos_i32 : i32 to index
      %abs_pos = arith.addi %start_pos, %s : index
      %logical_block = arith.divui %abs_pos, %block_size : index
      %pos_in_block = arith.remui %abs_pos, %block_size : index

      %physical_block_i32 = tensor.extract %block_tables_2d[%b, %logical_block] : tensor<?x?xi32>
      %physical_block = arith.index_cast %physical_block_i32 : i32 to index

      %new_v_slice = tensor.extract_slice %new_v[%b, %s, 0, 0] [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> to tensor<1x1x?x?x!elem_t>

      %new_v_3d = tensor.collapse_shape %new_v_slice [[0, 1], [2], [3]]
        : tensor<1x1x?x?x!elem_t> into tensor<1x?x?x!elem_t>

      %result = tensor.insert_slice %new_v_3d into %cache_v[%physical_block, %pos_in_block, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %result : tensor<?x?x?x?x!elem_t>
    }

    // Tie dimensions after loop
    %v_updated_tied = flow.tensor.reshape %v_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim} -> tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim}

    // Export updated V
    %v_updated_bv = hal.tensor.export %v_updated_tied : tensor<?x?x?x?x!elem_t>{%n_blocks, %cache_block_size, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Update list with new buffer views
    util.list.set %cache[%c0], %k_updated_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %v_updated_bv : !hal.buffer_view -> !util.list<?>

    util.return %cache : !util.list<?>
  }

}
