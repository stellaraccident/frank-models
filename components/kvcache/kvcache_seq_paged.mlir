// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sequence-Addressed Paged KV Cache with Host/Device Separation
//
// A paged KV cache where callers work with (seq_id, position) rather than
// block indices. The page table is managed internally ON THE HOST.
//
// Key design principle: Control data stays on host, compute data stays on device.
//   - seq_ids, positions, context_lens: !util.list<i32> (host)
//   - K/V blocks: tensor (device)
//   - page_table: !util.list<?> (host) - nested list of block indices per sequence
//
// Physical cache layout:
//   K/V blocks: [n_blocks, block_size, n_head_kv, head_dim]
//
// Cache object structure:
//   cache[0] = K blocks as !hal.buffer_view
//   cache[1] = V blocks as !hal.buffer_view
//   cache[2] = page_table as !util.list<?> - page_table[seq_id] = !util.list<i32>
//   cache[3] = block_size as i32
//
// Usage:
//   %cache = call @allocate(%n_seq, %max_seq_len, %n_head_kv, %head_dim, %block_size)
//   %k, %v = call @gather(%cache, %seq_ids, %context_lens, %total_tokens)
//   call @scatter_decode(%cache, %seq_ids, %positions, %new_k, %new_v)

!elem_t = f32

module @kvcache_seq_paged {

  // Allocate a sequence-addressed paged KV cache.
  //
  // Pre-allocates blocks for each sequence (simple strategy).
  // Sequence i gets blocks [i * blocks_per_seq, (i+1) * blocks_per_seq).
  //
  // Args:
  //   %n_seq:        Max concurrent sequences
  //   %max_seq_len:  Max tokens per sequence
  //   %n_head_kv:    Number of KV attention heads
  //   %head_dim:     Dimension per attention head
  //   %block_size:   Tokens per block
  //
  // Returns:
  //   !util.list<?> containing K blocks, V blocks, page_table (host), block_size
  util.func public @allocate(
      %n_seq: index,
      %max_seq_len: index,
      %n_head_kv: index,
      %head_dim: index,
      %block_size: index
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %element_size = util.sizeof !elem_t
    %affinity = arith.constant -1 : i64

    // Compute blocks per sequence: ceil(max_seq_len / block_size)
    %max_seq_len_minus_1 = arith.subi %max_seq_len, %c1 : index
    %blocks_per_seq = arith.divui %max_seq_len_minus_1, %block_size : index
    %blocks_per_seq_plus_1 = arith.addi %blocks_per_seq, %c1 : index

    // Total blocks = n_seq * blocks_per_seq
    %n_blocks = arith.muli %n_seq, %blocks_per_seq_plus_1 : index

    // Compute K/V buffer size
    %d0 = arith.muli %n_blocks, %block_size : index
    %d1 = arith.muli %d0, %n_head_kv : index
    %d2 = arith.muli %d1, %head_dim : index
    %kv_byte_size = arith.muli %d2, %element_size : index

    // Get device and allocator
    %device = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %memory_type = hal.memory_type<"DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|DispatchStorageRead|DispatchStorageWrite"> : i32

    // Allocate K blocks
    %k_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%kv_byte_size}
    %kv_element_type = hal.element_type<!elem_t> : i32
    %encoding_type = hal.encoding_type<dense_row_major> : i32
    %k_bv = hal.buffer_view.create buffer(%k_buffer : !hal.buffer)[%c0, %kv_byte_size]
                                   shape([%n_blocks, %block_size, %n_head_kv, %head_dim])
                                   type(%kv_element_type)
                                   encoding(%encoding_type) : !hal.buffer_view

    // Allocate V blocks
    %v_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%kv_byte_size}
    %v_bv = hal.buffer_view.create buffer(%v_buffer : !hal.buffer)[%c0, %kv_byte_size]
                                   shape([%n_blocks, %block_size, %n_head_kv, %head_dim])
                                   type(%kv_element_type)
                                   encoding(%encoding_type) : !hal.buffer_view

    // Create host-side page table: list of lists
    // page_table[seq_id] = list of physical block indices for that sequence
    %page_table = util.list.create %n_seq : !util.list<?>
    util.list.resize %page_table, %n_seq : !util.list<?>

    // Initialize page table: seq i gets blocks [i * blocks_per_seq, (i+1) * blocks_per_seq)
    scf.for %seq_id = %c0 to %n_seq step %c1 {
      %seq_blocks = util.list.create %blocks_per_seq_plus_1 : !util.list<i32>
      util.list.resize %seq_blocks, %blocks_per_seq_plus_1 : !util.list<i32>

      // Fill in physical block indices for this sequence
      %seq_offset = arith.muli %seq_id, %blocks_per_seq_plus_1 : index
      scf.for %block_idx = %c0 to %blocks_per_seq_plus_1 step %c1 {
        %physical_block = arith.addi %seq_offset, %block_idx : index
        %physical_block_i32 = arith.index_cast %physical_block : index to i32
        util.list.set %seq_blocks[%block_idx], %physical_block_i32 : i32 -> !util.list<i32>
      }

      util.list.set %page_table[%seq_id], %seq_blocks : !util.list<i32> -> !util.list<?>
    }

    // Store block_size
    %block_size_i32 = arith.index_cast %block_size : index to i32

    // Create cache list and store all components
    %this = util.list.create %c4 : !util.list<?>
    util.list.resize %this, %c4 : !util.list<?>
    util.list.set %this[%c0], %k_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %this[%c1], %v_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %this[%c2], %page_table : !util.list<?> -> !util.list<?>
    util.list.set %this[%c3], %block_size_i32 : i32 -> !util.list<?>

    util.return %this : !util.list<?>
  }

  // Gather K and V for attention using sequence IDs and context lengths.
  //
  // All control data (seq_ids, context_lens) is host-side. Page table lookups
  // happen on host, then flat block_indices are passed to device for gather.
  //
  // Args:
  //   %cache:        Cache object from @allocate
  //   %seq_ids:      !util.list<i32> [batch] which sequences to gather from (host)
  //   %context_lens: !util.list<i32> [batch] context length per sequence (host)
  //   %total_tokens: sum(context_lens), needed for output shape
  //
  // Returns:
  //   %k: [total_tokens, n_head_kv, head_dim]
  //   %v: [total_tokens, n_head_kv, head_dim]
  util.func public @gather(
      %cache: !util.list<?>,
      %seq_ids: !util.list<i32>,
      %context_lens: !util.list<i32>,
      %total_tokens: index
  ) -> (tensor<?x?x?x!elem_t>, tensor<?x?x?x!elem_t>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Get batch size from seq_ids list
    %batch_size = util.list.size %seq_ids : !util.list<i32>

    // Import K blocks
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %n_blocks = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %block_size = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %n_head_kv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %head_dim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[3] : index
    %k_blocks = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Import V blocks
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %v_blocks = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Get page table (host-side)
    %page_table = util.list.get %cache[%c2] : !util.list<?> -> !util.list<?>

    // Build flat block_indices on host
    // block_indices[out_idx] = physical block for token out_idx
    %block_indices_list = util.list.create %total_tokens : !util.list<i32>
    util.list.resize %block_indices_list, %total_tokens : !util.list<i32>

    // Loop over batch and positions to fill block_indices
    %final_out_idx = scf.for %b = %c0 to %batch_size step %c1
        iter_args(%out_idx = %c0) -> (index) {
      %seq_id_i32 = util.list.get %seq_ids[%b] : !util.list<i32> -> i32
      %seq_id = arith.index_cast %seq_id_i32 : i32 to index
      %ctx_len_i32 = util.list.get %context_lens[%b] : !util.list<i32> -> i32
      %ctx_len = arith.index_cast %ctx_len_i32 : i32 to index

      // Get this sequence's block list
      %seq_blocks = util.list.get %page_table[%seq_id] : !util.list<?> -> !util.list<i32>

      // Loop over positions in this sequence
      %next_out_idx = scf.for %pos = %c0 to %ctx_len step %c1
          iter_args(%curr_out_idx = %out_idx) -> (index) {
        %logical_block = arith.divui %pos, %block_size : index
        %physical_block = util.list.get %seq_blocks[%logical_block] : !util.list<i32> -> i32

        util.list.set %block_indices_list[%curr_out_idx], %physical_block : i32 -> !util.list<i32>

        %next = arith.addi %curr_out_idx, %c1 : index
        scf.yield %next : index
      }

      scf.yield %next_out_idx : index
    }

    // Convert block_indices list to tensor for device
    %block_indices_init = tensor.empty(%total_tokens) : tensor<?xi32>
    %block_indices = scf.for %i = %c0 to %total_tokens step %c1
        iter_args(%acc = %block_indices_init) -> (tensor<?xi32>) {
      %val = util.list.get %block_indices_list[%i] : !util.list<i32> -> i32
      %updated = tensor.insert %val into %acc[%i] : tensor<?xi32>
      scf.yield %updated : tensor<?xi32>
    }

    // Allocate output tensors
    %k_init = tensor.empty(%total_tokens, %n_head_kv, %head_dim) : tensor<?x?x?x!elem_t>
    %v_init = tensor.empty(%total_tokens, %n_head_kv, %head_dim) : tensor<?x?x?x!elem_t>

    // Gather K using linalg.generic (device-side)
    %k_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(tok, head, dim) -> (tok)>,
        affine_map<(tok, head, dim) -> (tok, head, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%block_indices : tensor<?xi32>)
      outs(%k_init : tensor<?x?x?x!elem_t>) {
    ^bb0(%block_idx_i32: i32, %out: !elem_t):
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %tok_idx = linalg.index 0 : index
      %head_idx = linalg.index 1 : index
      %dim_idx = linalg.index 2 : index

      %pos_in_block = arith.remui %tok_idx, %block_size : index
      %k_val = tensor.extract %k_blocks[%block_idx, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>

      linalg.yield %k_val : !elem_t
    } -> tensor<?x?x?x!elem_t>

    // Gather V similarly
    %v_gathered = linalg.generic {
      indexing_maps = [
        affine_map<(tok, head, dim) -> (tok)>,
        affine_map<(tok, head, dim) -> (tok, head, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%block_indices : tensor<?xi32>)
      outs(%v_init : tensor<?x?x?x!elem_t>) {
    ^bb0(%block_idx_i32: i32, %out: !elem_t):
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %tok_idx = linalg.index 0 : index
      %head_idx = linalg.index 1 : index
      %dim_idx = linalg.index 2 : index

      %pos_in_block = arith.remui %tok_idx, %block_size : index
      %v_val = tensor.extract %v_blocks[%block_idx, %pos_in_block, %head_idx, %dim_idx]
        : tensor<?x?x?x?x!elem_t>

      linalg.yield %v_val : !elem_t
    } -> tensor<?x?x?x!elem_t>

    util.return %k_gathered, %v_gathered : tensor<?x?x?x!elem_t>, tensor<?x?x?x!elem_t>
  }

  // Scatter one token per sequence (decode phase).
  //
  // All control data (seq_ids, positions) is host-side. Page table lookups
  // happen on host, then scatter uses computed block indices.
  //
  // Mutates cache in place.
  //
  // Args:
  //   %cache:     Cache object from @allocate
  //   %seq_ids:   !util.list<i32> [batch] which sequences to write to (host)
  //   %positions: !util.list<i32> [batch] write position per sequence (host)
  //   %new_k:     tensor [batch, n_head_kv, head_dim]
  //   %new_v:     tensor [batch, n_head_kv, head_dim]
  util.func public @scatter_decode(
      %cache: !util.list<?>,
      %seq_ids: !util.list<i32>,
      %positions: !util.list<i32>,
      %new_k: tensor<?x?x?x!elem_t>,
      %new_v: tensor<?x?x?x!elem_t>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Get batch size and dimensions
    %batch_size = tensor.dim %new_k, %c0 : tensor<?x?x?x!elem_t>
    %n_head_kv = tensor.dim %new_k, %c1 : tensor<?x?x?x!elem_t>
    %head_dim = tensor.dim %new_k, %c2 : tensor<?x?x?x!elem_t>

    // Import K blocks
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %n_blocks = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %block_size = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %k_blocks = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Import V blocks
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %v_blocks = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}

    // Get page table (host-side)
    %page_table = util.list.get %cache[%c2] : !util.list<?> -> !util.list<?>

    // Build block_indices and pos_in_blocks on host
    %block_indices_list = util.list.create %batch_size : !util.list<i32>
    util.list.resize %block_indices_list, %batch_size : !util.list<i32>
    %pos_in_blocks_list = util.list.create %batch_size : !util.list<i32>
    util.list.resize %pos_in_blocks_list, %batch_size : !util.list<i32>

    scf.for %i = %c0 to %batch_size step %c1 {
      %seq_id_i32 = util.list.get %seq_ids[%i] : !util.list<i32> -> i32
      %seq_id = arith.index_cast %seq_id_i32 : i32 to index
      %position_i32 = util.list.get %positions[%i] : !util.list<i32> -> i32
      %position = arith.index_cast %position_i32 : i32 to index

      %logical_block = arith.divui %position, %block_size : index
      %pos_in_block = arith.remui %position, %block_size : index
      %pos_in_block_i32 = arith.index_cast %pos_in_block : index to i32

      // Host-side page table lookup
      %seq_blocks = util.list.get %page_table[%seq_id] : !util.list<?> -> !util.list<i32>
      %physical_block = util.list.get %seq_blocks[%logical_block] : !util.list<i32> -> i32

      util.list.set %block_indices_list[%i], %physical_block : i32 -> !util.list<i32>
      util.list.set %pos_in_blocks_list[%i], %pos_in_block_i32 : i32 -> !util.list<i32>
    }

    // Convert to device tensors
    %block_indices_init = tensor.empty(%batch_size) : tensor<?xi32>
    %block_indices = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%acc = %block_indices_init) -> (tensor<?xi32>) {
      %val = util.list.get %block_indices_list[%i] : !util.list<i32> -> i32
      %updated = tensor.insert %val into %acc[%i] : tensor<?xi32>
      scf.yield %updated : tensor<?xi32>
    }

    %pos_in_blocks_init = tensor.empty(%batch_size) : tensor<?xi32>
    %pos_in_blocks = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%acc = %pos_in_blocks_init) -> (tensor<?xi32>) {
      %val = util.list.get %pos_in_blocks_list[%i] : !util.list<i32> -> i32
      %updated = tensor.insert %val into %acc[%i] : tensor<?xi32>
      scf.yield %updated : tensor<?xi32>
    }

    // Scatter K: loop over batch using device tensors
    %k_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%k_cache = %k_blocks) -> (tensor<?x?x?x?x!elem_t>) {
      %block_idx_i32 = tensor.extract %block_indices[%i] : tensor<?xi32>
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %pos_i32 = tensor.extract %pos_in_blocks[%i] : tensor<?xi32>
      %pos = arith.index_cast %pos_i32 : i32 to index

      %new_k_slice = tensor.extract_slice %new_k[%i, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?x!elem_t> to tensor<1x?x?x!elem_t>

      %updated = tensor.insert_slice %new_k_slice into %k_cache[%block_idx, %pos, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %updated : tensor<?x?x?x?x!elem_t>
    }

    // Tie dimensions and export K
    %k_updated_tied = flow.tensor.reshape %k_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}
    %k_updated_bv = hal.tensor.export %k_updated_tied : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Scatter V similarly
    %v_updated = scf.for %i = %c0 to %batch_size step %c1
        iter_args(%v_cache = %v_blocks) -> (tensor<?x?x?x?x!elem_t>) {
      %block_idx_i32 = tensor.extract %block_indices[%i] : tensor<?xi32>
      %block_idx = arith.index_cast %block_idx_i32 : i32 to index
      %pos_i32 = tensor.extract %pos_in_blocks[%i] : tensor<?xi32>
      %pos = arith.index_cast %pos_i32 : i32 to index

      %new_v_slice = tensor.extract_slice %new_v[%i, 0, 0] [1, %n_head_kv, %head_dim] [1, 1, 1]
        : tensor<?x?x?x!elem_t> to tensor<1x?x?x!elem_t>

      %updated = tensor.insert_slice %new_v_slice into %v_cache[%block_idx, %pos, 0, 0]
        [1, 1, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<1x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>

      scf.yield %updated : tensor<?x?x?x?x!elem_t>
    }

    // Tie dimensions and export V
    %v_updated_tied = flow.tensor.reshape %v_updated : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim}
    %v_updated_bv = hal.tensor.export %v_updated_tied : tensor<?x?x?x?x!elem_t>{%n_blocks, %block_size, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Update cache in place
    util.list.set %cache[%c0], %k_updated_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %v_updated_bv : !hal.buffer_view -> !util.list<?>

    util.return
  }

}
