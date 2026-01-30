// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct KV Cache - batched cache for a set of sequences.
//
// Dimension layout (all tensors):
//   dim 0: batch_size   - Number of sequences in the batch
//   dim 1: seq_len      - Sequence position (max_seq_len for cache storage)
//   dim 2: n_head_kv    - Number of KV heads (may differ from Q heads in GQA)
//   dim 3: head_dim     - Dimension per attention head
//
// Cache object structure (internal):
//   this[0] = K cache as !hal.buffer_view [batch_size, max_seq_len, n_head_kv, head_dim]
//   this[1] = V cache as !hal.buffer_view [batch_size, max_seq_len, n_head_kv, head_dim]
//
// Public API uses tensor<?x?x?x?x!elem_t> types at the boundary.
//
// Usage:
//   %cache = call @allocate(%batch_size, %max_seq_len, %n_head_kv, %head_dim)
//   %k, %v = call @gather(%cache, %seq_len)
//   %updated = call @scatter(%cache, %new_k, %new_v, %write_pos)

// Cache element type - change here to specialize (e.g., f16, bf16)
!elem_t = f32

module @kvcache_components {

  // Allocate a new batched KV cache initialized to zeros.
  //
  // Args:
  //   %batch_size:   Number of sequences in batch
  //   %max_seq_len:  Maximum sequence length (cache capacity)
  //   %n_head_kv:    Number of KV attention heads
  //   %head_dim:     Dimension per attention head
  //
  // Returns:
  //   !util.list<?> containing K and V caches, each [batch_size, max_seq_len, n_head_kv, head_dim]
  util.func public @allocate(
      %batch_size: index,
      %max_seq_len: index,
      %n_head_kv: index,
      %head_dim: index
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %element_size = util.sizeof !elem_t
    %affinity = arith.constant -1 : i64

    // Compute buffer size: batch_size * max_seq_len * n_head_kv * head_dim * sizeof(element)
    %d0 = arith.muli %batch_size, %max_seq_len : index
    %d1 = arith.muli %d0, %n_head_kv : index
    %d2 = arith.muli %d1, %head_dim : index
    %byte_size = arith.muli %d2, %element_size : index

    // Get device and allocator
    %device = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator

    // Allocate K buffer
    %memory_type = hal.memory_type<"DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|DispatchStorageRead|DispatchStorageWrite"> : i32
    %k_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%byte_size}

    // Create K buffer view [batch_size, max_seq_len, n_head_kv, head_dim]
    %element_type = hal.element_type<!elem_t> : i32
    %encoding_type = hal.encoding_type<dense_row_major> : i32
    %k_bv = hal.buffer_view.create buffer(%k_buffer : !hal.buffer)[%c0, %byte_size]
                                   shape([%batch_size, %max_seq_len, %n_head_kv, %head_dim])
                                   type(%element_type)
                                   encoding(%encoding_type) : !hal.buffer_view

    // Allocate V buffer (distinct allocation)
    %v_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
        affinity(%affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%byte_size}

    // Create V buffer view
    %v_bv = hal.buffer_view.create buffer(%v_buffer : !hal.buffer)[%c0, %byte_size]
                                   shape([%batch_size, %max_seq_len, %n_head_kv, %head_dim])
                                   type(%element_type)
                                   encoding(%encoding_type) : !hal.buffer_view

    // Create list and store buffer views
    %this = util.list.create %c2 : !util.list<?>
    util.list.resize %this, %c2 : !util.list<?>
    util.list.set %this[%c0], %k_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %this[%c1], %v_bv : !hal.buffer_view -> !util.list<?>

    util.return %this : !util.list<?>
  }

  // Gather K and V from cache for positions [0, seq_len) for all batch elements.
  //
  // Args:
  //   %this:    Cache object from @allocate
  //   %seq_len: Number of positions to gather (same for all batch elements)
  //
  // Returns:
  //   %k: [batch_size, seq_len, n_head_kv, head_dim]
  //   %v: [batch_size, seq_len, n_head_kv, head_dim]
  util.func public @gather(
      %this: !util.list<?>,
      %seq_len: index
  ) -> (tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Get K buffer view and extract dimensions
    %k_bv = util.list.get %this[%c0] : !util.list<?> -> !hal.buffer_view
    %batch_size = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %max_seq_len = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %n_head_kv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %head_dim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[3] : index

    // Import K cache [batch_size, max_seq_len, n_head_kv, head_dim]
    %k_cache = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%batch_size, %max_seq_len, %n_head_kv, %head_dim}

    // Extract slice [batch_size, 0:seq_len, n_head_kv, head_dim]
    %k_result = tensor.extract_slice %k_cache[0, 0, 0, 0] [%batch_size, %seq_len, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> to tensor<?x?x?x?x!elem_t>

    // Get V buffer view and import
    %v_bv = util.list.get %this[%c1] : !util.list<?> -> !hal.buffer_view
    %v_cache = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%batch_size, %max_seq_len, %n_head_kv, %head_dim}

    // Extract slice [batch_size, 0:seq_len, n_head_kv, head_dim]
    %v_result = tensor.extract_slice %v_cache[0, 0, 0, 0] [%batch_size, %seq_len, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> to tensor<?x?x?x?x!elem_t>

    util.return %k_result, %v_result : tensor<?x?x?x?x!elem_t>, tensor<?x?x?x?x!elem_t>
  }

  // Scatter new K and V to cache at write_pos for all batch elements.
  //
  // Args:
  //   %this:      Cache object from @allocate
  //   %new_k:     [batch_size, new_tokens, n_head_kv, head_dim] - new K values to write
  //   %new_v:     [batch_size, new_tokens, n_head_kv, head_dim] - new V values to write
  //   %write_pos: Starting sequence position to write (same for all batch elements)
  //
  // Returns:
  //   Updated cache object
  util.func public @scatter(
      %this: !util.list<?>,
      %new_k: tensor<?x?x?x?x!elem_t>,
      %new_v: tensor<?x?x?x?x!elem_t>,
      %write_pos: index
  ) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Get dimensions from new_k [batch_size, new_tokens, n_head_kv, head_dim]
    %batch_size = tensor.dim %new_k, %c0 : tensor<?x?x?x?x!elem_t>
    %new_tokens = tensor.dim %new_k, %c1 : tensor<?x?x?x?x!elem_t>
    %n_head_kv = tensor.dim %new_k, %c2 : tensor<?x?x?x?x!elem_t>
    %head_dim = tensor.dim %new_k, %c3 : tensor<?x?x?x?x!elem_t>

    // Get K buffer view and import [batch_size, max_seq_len, n_head_kv, head_dim]
    %k_bv = util.list.get %this[%c0] : !util.list<?> -> !hal.buffer_view
    %max_seq_len = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %k_cache = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%batch_size, %max_seq_len, %n_head_kv, %head_dim}

    // Insert new_k at [0, write_pos, 0, 0] with size [batch_size, new_tokens, n_head_kv, head_dim]
    %k_updated = tensor.insert_slice %new_k into %k_cache[0, %write_pos, 0, 0]
        [%batch_size, %new_tokens, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>
    %k_updated_bv = hal.tensor.export %k_updated : tensor<?x?x?x?x!elem_t>{%batch_size, %max_seq_len, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Get V buffer view and import
    %v_bv = util.list.get %this[%c1] : !util.list<?> -> !hal.buffer_view
    %v_cache = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?x?x!elem_t>{%batch_size, %max_seq_len, %n_head_kv, %head_dim}

    // Insert new_v at [0, write_pos, 0, 0] with size [batch_size, new_tokens, n_head_kv, head_dim]
    %v_updated = tensor.insert_slice %new_v into %v_cache[0, %write_pos, 0, 0]
        [%batch_size, %new_tokens, %n_head_kv, %head_dim] [1, 1, 1, 1]
        : tensor<?x?x?x?x!elem_t> into tensor<?x?x?x?x!elem_t>
    %v_updated_bv = hal.tensor.export %v_updated : tensor<?x?x?x?x!elem_t>{%batch_size, %max_seq_len, %n_head_kv, %head_dim} -> !hal.buffer_view

    // Update list with new buffer views
    util.list.set %this[%c0], %k_updated_bv : !hal.buffer_view -> !util.list<?>
    util.list.set %this[%c1], %v_updated_bv : !hal.buffer_view -> !util.list<?>

    util.return %this : !util.list<?>
  }

}
