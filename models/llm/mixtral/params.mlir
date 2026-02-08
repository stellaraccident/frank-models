// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mixtral-8x7B parameter accessors.
//
// Uses flow.tensor.constant with #flow.parameter.named to load parameters.
// Layer dispatch uses nested scf.if - this is verbose but works.
// Future: Use IREE's dynamic parameter key support (PR #23426) for cleaner code.
//
// Shapes (Mixtral-8x7B):
//   vocab_size=32000, n_embd=4096, n_head=32, n_head_kv=8, head_dim=128
//   n_ff=14336, n_expert=8, n_layers=32

module @model_params {

  // ===== Model-level parameters =====

  util.func public @token_embd_weight() -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"token_embd.weight"> : tensor<32000x4096xf32>
    %dyn = tensor.cast %w : tensor<32000x4096xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @output_norm_weight() -> tensor<?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output_norm.weight"> : tensor<4096xf32>
    %dyn = tensor.cast %w : tensor<4096xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @output_weight() -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"output.weight"> : tensor<4096x32000xf32>
    %dyn = tensor.cast %w : tensor<4096x32000xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  // ===== Layer-level parameters =====
  // Dispatch to correct layer using nested scf.if (verbose but works).

  // --- attn_norm_weight ---
  util.func public @attn_norm_weight(%layer: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c9 = arith.constant 9 : i32
    %c10 = arith.constant 10 : i32
    %c11 = arith.constant 11 : i32
    %c12 = arith.constant 12 : i32
    %c13 = arith.constant 13 : i32
    %c14 = arith.constant 14 : i32
    %c15 = arith.constant 15 : i32
    %c16 = arith.constant 16 : i32
    %c17 = arith.constant 17 : i32
    %c18 = arith.constant 18 : i32
    %c19 = arith.constant 19 : i32
    %c20 = arith.constant 20 : i32
    %c21 = arith.constant 21 : i32
    %c22 = arith.constant 22 : i32
    %c23 = arith.constant 23 : i32
    %c24 = arith.constant 24 : i32
    %c25 = arith.constant 25 : i32
    %c26 = arith.constant 26 : i32
    %c27 = arith.constant 27 : i32
    %c28 = arith.constant 28 : i32
    %c29 = arith.constant 29 : i32
    %c30 = arith.constant 30 : i32
    %c31 = arith.constant 31 : i32

    %is0 = arith.cmpi eq, %layer, %c0 : i32
    %w = scf.if %is0 -> (tensor<4096xf32>) {
      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_norm.weight"> : tensor<4096xf32>
      scf.yield %t : tensor<4096xf32>
    } else {
      %is1 = arith.cmpi eq, %layer, %c1 : i32
      %r1 = scf.if %is1 -> (tensor<4096xf32>) {
        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.1.attn_norm.weight"> : tensor<4096xf32>
        scf.yield %t : tensor<4096xf32>
      } else {
        %is2 = arith.cmpi eq, %layer, %c2 : i32
        %r2 = scf.if %is2 -> (tensor<4096xf32>) {
          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.2.attn_norm.weight"> : tensor<4096xf32>
          scf.yield %t : tensor<4096xf32>
        } else {
          %is3 = arith.cmpi eq, %layer, %c3 : i32
          %r3 = scf.if %is3 -> (tensor<4096xf32>) {
            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.3.attn_norm.weight"> : tensor<4096xf32>
            scf.yield %t : tensor<4096xf32>
          } else {
            %is4 = arith.cmpi eq, %layer, %c4 : i32
            %r4 = scf.if %is4 -> (tensor<4096xf32>) {
              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.4.attn_norm.weight"> : tensor<4096xf32>
              scf.yield %t : tensor<4096xf32>
            } else {
              %is5 = arith.cmpi eq, %layer, %c5 : i32
              %r5 = scf.if %is5 -> (tensor<4096xf32>) {
                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.5.attn_norm.weight"> : tensor<4096xf32>
                scf.yield %t : tensor<4096xf32>
              } else {
                %is6 = arith.cmpi eq, %layer, %c6 : i32
                %r6 = scf.if %is6 -> (tensor<4096xf32>) {
                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.6.attn_norm.weight"> : tensor<4096xf32>
                  scf.yield %t : tensor<4096xf32>
                } else {
                  %is7 = arith.cmpi eq, %layer, %c7 : i32
                  %r7 = scf.if %is7 -> (tensor<4096xf32>) {
                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.7.attn_norm.weight"> : tensor<4096xf32>
                    scf.yield %t : tensor<4096xf32>
                  } else {
                    %is8 = arith.cmpi eq, %layer, %c8 : i32
                    %r8 = scf.if %is8 -> (tensor<4096xf32>) {
                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.8.attn_norm.weight"> : tensor<4096xf32>
                      scf.yield %t : tensor<4096xf32>
                    } else {
                      %is9 = arith.cmpi eq, %layer, %c9 : i32
                      %r9 = scf.if %is9 -> (tensor<4096xf32>) {
                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.9.attn_norm.weight"> : tensor<4096xf32>
                        scf.yield %t : tensor<4096xf32>
                      } else {
                        %is10 = arith.cmpi eq, %layer, %c10 : i32
                        %r10 = scf.if %is10 -> (tensor<4096xf32>) {
                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.10.attn_norm.weight"> : tensor<4096xf32>
                          scf.yield %t : tensor<4096xf32>
                        } else {
                          %is11 = arith.cmpi eq, %layer, %c11 : i32
                          %r11 = scf.if %is11 -> (tensor<4096xf32>) {
                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.11.attn_norm.weight"> : tensor<4096xf32>
                            scf.yield %t : tensor<4096xf32>
                          } else {
                            %is12 = arith.cmpi eq, %layer, %c12 : i32
                            %r12 = scf.if %is12 -> (tensor<4096xf32>) {
                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.12.attn_norm.weight"> : tensor<4096xf32>
                              scf.yield %t : tensor<4096xf32>
                            } else {
                              %is13 = arith.cmpi eq, %layer, %c13 : i32
                              %r13 = scf.if %is13 -> (tensor<4096xf32>) {
                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.13.attn_norm.weight"> : tensor<4096xf32>
                                scf.yield %t : tensor<4096xf32>
                              } else {
                                %is14 = arith.cmpi eq, %layer, %c14 : i32
                                %r14 = scf.if %is14 -> (tensor<4096xf32>) {
                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.14.attn_norm.weight"> : tensor<4096xf32>
                                  scf.yield %t : tensor<4096xf32>
                                } else {
                                  %is15 = arith.cmpi eq, %layer, %c15 : i32
                                  %r15 = scf.if %is15 -> (tensor<4096xf32>) {
                                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.15.attn_norm.weight"> : tensor<4096xf32>
                                    scf.yield %t : tensor<4096xf32>
                                  } else {
                                    %is16 = arith.cmpi eq, %layer, %c16 : i32
                                    %r16 = scf.if %is16 -> (tensor<4096xf32>) {
                                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.16.attn_norm.weight"> : tensor<4096xf32>
                                      scf.yield %t : tensor<4096xf32>
                                    } else {
                                      %is17 = arith.cmpi eq, %layer, %c17 : i32
                                      %r17 = scf.if %is17 -> (tensor<4096xf32>) {
                                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.17.attn_norm.weight"> : tensor<4096xf32>
                                        scf.yield %t : tensor<4096xf32>
                                      } else {
                                        %is18 = arith.cmpi eq, %layer, %c18 : i32
                                        %r18 = scf.if %is18 -> (tensor<4096xf32>) {
                                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.18.attn_norm.weight"> : tensor<4096xf32>
                                          scf.yield %t : tensor<4096xf32>
                                        } else {
                                          %is19 = arith.cmpi eq, %layer, %c19 : i32
                                          %r19 = scf.if %is19 -> (tensor<4096xf32>) {
                                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.19.attn_norm.weight"> : tensor<4096xf32>
                                            scf.yield %t : tensor<4096xf32>
                                          } else {
                                            %is20 = arith.cmpi eq, %layer, %c20 : i32
                                            %r20 = scf.if %is20 -> (tensor<4096xf32>) {
                                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.20.attn_norm.weight"> : tensor<4096xf32>
                                              scf.yield %t : tensor<4096xf32>
                                            } else {
                                              %is21 = arith.cmpi eq, %layer, %c21 : i32
                                              %r21 = scf.if %is21 -> (tensor<4096xf32>) {
                                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.21.attn_norm.weight"> : tensor<4096xf32>
                                                scf.yield %t : tensor<4096xf32>
                                              } else {
                                                %is22 = arith.cmpi eq, %layer, %c22 : i32
                                                %r22 = scf.if %is22 -> (tensor<4096xf32>) {
                                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.22.attn_norm.weight"> : tensor<4096xf32>
                                                  scf.yield %t : tensor<4096xf32>
                                                } else {
                                                  %is23 = arith.cmpi eq, %layer, %c23 : i32
                                                  %r23 = scf.if %is23 -> (tensor<4096xf32>) {
                                                    %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.23.attn_norm.weight"> : tensor<4096xf32>
                                                    scf.yield %t : tensor<4096xf32>
                                                  } else {
                                                    %is24 = arith.cmpi eq, %layer, %c24 : i32
                                                    %r24 = scf.if %is24 -> (tensor<4096xf32>) {
                                                      %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.24.attn_norm.weight"> : tensor<4096xf32>
                                                      scf.yield %t : tensor<4096xf32>
                                                    } else {
                                                      %is25 = arith.cmpi eq, %layer, %c25 : i32
                                                      %r25 = scf.if %is25 -> (tensor<4096xf32>) {
                                                        %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.25.attn_norm.weight"> : tensor<4096xf32>
                                                        scf.yield %t : tensor<4096xf32>
                                                      } else {
                                                        %is26 = arith.cmpi eq, %layer, %c26 : i32
                                                        %r26 = scf.if %is26 -> (tensor<4096xf32>) {
                                                          %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.26.attn_norm.weight"> : tensor<4096xf32>
                                                          scf.yield %t : tensor<4096xf32>
                                                        } else {
                                                          %is27 = arith.cmpi eq, %layer, %c27 : i32
                                                          %r27 = scf.if %is27 -> (tensor<4096xf32>) {
                                                            %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.27.attn_norm.weight"> : tensor<4096xf32>
                                                            scf.yield %t : tensor<4096xf32>
                                                          } else {
                                                            %is28 = arith.cmpi eq, %layer, %c28 : i32
                                                            %r28 = scf.if %is28 -> (tensor<4096xf32>) {
                                                              %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.28.attn_norm.weight"> : tensor<4096xf32>
                                                              scf.yield %t : tensor<4096xf32>
                                                            } else {
                                                              %is29 = arith.cmpi eq, %layer, %c29 : i32
                                                              %r29 = scf.if %is29 -> (tensor<4096xf32>) {
                                                                %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.29.attn_norm.weight"> : tensor<4096xf32>
                                                                scf.yield %t : tensor<4096xf32>
                                                              } else {
                                                                %is30 = arith.cmpi eq, %layer, %c30 : i32
                                                                %r30 = scf.if %is30 -> (tensor<4096xf32>) {
                                                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.30.attn_norm.weight"> : tensor<4096xf32>
                                                                  scf.yield %t : tensor<4096xf32>
                                                                } else {
                                                                  %t = flow.tensor.constant #flow.parameter.named<"model"::"blk.31.attn_norm.weight"> : tensor<4096xf32>
                                                                  scf.yield %t : tensor<4096xf32>
                                                                }
                                                                scf.yield %r30 : tensor<4096xf32>
                                                              }
                                                              scf.yield %r29 : tensor<4096xf32>
                                                            }
                                                            scf.yield %r28 : tensor<4096xf32>
                                                          }
                                                          scf.yield %r27 : tensor<4096xf32>
                                                        }
                                                        scf.yield %r26 : tensor<4096xf32>
                                                      }
                                                      scf.yield %r25 : tensor<4096xf32>
                                                    }
                                                    scf.yield %r24 : tensor<4096xf32>
                                                  }
                                                  scf.yield %r23 : tensor<4096xf32>
                                                }
                                                scf.yield %r22 : tensor<4096xf32>
                                              }
                                              scf.yield %r21 : tensor<4096xf32>
                                            }
                                            scf.yield %r20 : tensor<4096xf32>
                                          }
                                          scf.yield %r19 : tensor<4096xf32>
                                        }
                                        scf.yield %r18 : tensor<4096xf32>
                                      }
                                      scf.yield %r17 : tensor<4096xf32>
                                    }
                                    scf.yield %r16 : tensor<4096xf32>
                                  }
                                  scf.yield %r15 : tensor<4096xf32>
                                }
                                scf.yield %r14 : tensor<4096xf32>
                              }
                              scf.yield %r13 : tensor<4096xf32>
                            }
                            scf.yield %r12 : tensor<4096xf32>
                          }
                          scf.yield %r11 : tensor<4096xf32>
                        }
                        scf.yield %r10 : tensor<4096xf32>
                      }
                      scf.yield %r9 : tensor<4096xf32>
                    }
                    scf.yield %r8 : tensor<4096xf32>
                  }
                  scf.yield %r7 : tensor<4096xf32>
                }
                scf.yield %r6 : tensor<4096xf32>
              }
              scf.yield %r5 : tensor<4096xf32>
            }
            scf.yield %r4 : tensor<4096xf32>
          }
          scf.yield %r3 : tensor<4096xf32>
        }
        scf.yield %r2 : tensor<4096xf32>
      }
      scf.yield %r1 : tensor<4096xf32>
    }
    %dyn = tensor.cast %w : tensor<4096xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  // The remaining layer accessors follow the same pattern.
  // For brevity in this initial version, we'll generate them with a Python script.
  // TODO: Add remaining accessors or generate this file programmatically.

  // For now, stub out the remaining functions to allow linking to succeed.
  // These will be properly implemented when we switch to dynamic parameter keys.

  util.func public @ffn_norm_weight(%layer: i32) -> tensor<?xf32> {
    // Placeholder - same pattern as attn_norm_weight
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_norm.weight"> : tensor<4096xf32>
    %dyn = tensor.cast %w : tensor<4096xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @attn_q_weight(%layer: i32) -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_q.weight"> : tensor<4096x4096xf32>
    %dyn = tensor.cast %w : tensor<4096x4096xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @attn_k_weight(%layer: i32) -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_k.weight"> : tensor<4096x1024xf32>
    %dyn = tensor.cast %w : tensor<4096x1024xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @attn_v_weight(%layer: i32) -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_v.weight"> : tensor<4096x1024xf32>
    %dyn = tensor.cast %w : tensor<4096x1024xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @attn_output_weight(%layer: i32) -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_output.weight"> : tensor<4096x4096xf32>
    %dyn = tensor.cast %w : tensor<4096x4096xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @attn_q_bias(%layer: i32) -> tensor<?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_q.bias"> : tensor<4096xf32>
    %dyn = tensor.cast %w : tensor<4096xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @attn_k_bias(%layer: i32) -> tensor<?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_k.bias"> : tensor<1024xf32>
    %dyn = tensor.cast %w : tensor<1024xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @attn_v_bias(%layer: i32) -> tensor<?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_v.bias"> : tensor<1024xf32>
    %dyn = tensor.cast %w : tensor<1024xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @attn_output_bias(%layer: i32) -> tensor<?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.attn_output.bias"> : tensor<4096xf32>
    %dyn = tensor.cast %w : tensor<4096xf32> to tensor<?xf32>
    util.return %dyn : tensor<?xf32>
  }

  util.func public @ffn_gate_inp_weight(%layer: i32) -> tensor<?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_gate_inp.weight"> : tensor<8x4096xf32>
    %dyn = tensor.cast %w : tensor<8x4096xf32> to tensor<?x?xf32>
    util.return %dyn : tensor<?x?xf32>
  }

  util.func public @ffn_up_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_up_exps.weight"> : tensor<14336x4096x8xf32>
    %dyn = tensor.cast %w : tensor<14336x4096x8xf32> to tensor<?x?x?xf32>
    util.return %dyn : tensor<?x?x?xf32>
  }

  util.func public @ffn_gate_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_gate_exps.weight"> : tensor<14336x4096x8xf32>
    %dyn = tensor.cast %w : tensor<14336x4096x8xf32> to tensor<?x?x?xf32>
    util.return %dyn : tensor<?x?x?xf32>
  }

  util.func public @ffn_down_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %w = flow.tensor.constant #flow.parameter.named<"model"::"blk.0.ffn_down_exps.weight"> : tensor<4096x14336x8xf32>
    %dyn = tensor.cast %w : tensor<4096x14336x8xf32> to tensor<?x?x?xf32>
    util.return %dyn : tensor<?x?x?xf32>
  }

}
