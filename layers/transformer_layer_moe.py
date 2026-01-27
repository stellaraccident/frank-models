"""Specialized module generator for transformer_layer_moe.

Generates a combined MLIR module that includes:
- Parameter accessor functions using flow.tensor.constant
- The transformer layer function (inlined from template)
- Static-shaped @forward entry point

Only compute component imports remain external (resolved by iree-link).

Also provides helpers for creating random parameter sets for testing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MoELayerConfig:
    """Configuration for a single MoE transformer layer."""

    # Model dimensions
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_ff: int
    n_expert: int
    n_expert_used: int

    # Static shape for test wrapper
    batch: int = 1
    seq_len: int = 4

    # Layer identity
    layer_idx: int = 0

    # Hyperparameters
    rms_eps: float = 1e-5
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    use_bias: bool = False
    normalize_weights: bool = False

    @property
    def n_embd_kv(self) -> int:
        return self.n_head_kv * self.head_dim


# GGUF-style parameter names and their shapes for a given config.
def _param_specs(cfg: MoELayerConfig) -> list[tuple[str, tuple[int, ...]]]:
    """Return (gguf_name, shape) for all layer parameters."""
    idx = cfg.layer_idx
    kv = cfg.n_embd_kv
    return [
        # Normalization
        (f"blk.{idx}.attn_norm.weight", (cfg.n_embd,)),
        (f"blk.{idx}.ffn_norm.weight", (cfg.n_embd,)),
        # Attention projections
        (f"blk.{idx}.attn_q.weight", (cfg.n_embd, cfg.n_embd)),
        (f"blk.{idx}.attn_k.weight", (cfg.n_embd, kv)),
        (f"blk.{idx}.attn_v.weight", (cfg.n_embd, kv)),
        (f"blk.{idx}.attn_output.weight", (cfg.n_embd, cfg.n_embd)),
        # Attention biases (zeros when not used, still needed for interface)
        (f"blk.{idx}.attn_q.bias", (cfg.n_embd,)),
        (f"blk.{idx}.attn_k.bias", (kv,)),
        (f"blk.{idx}.attn_v.bias", (kv,)),
        (f"blk.{idx}.attn_output.bias", (cfg.n_embd,)),
        # MoE
        (f"blk.{idx}.ffn_gate_inp.weight", (cfg.n_expert, cfg.n_embd)),
        (f"blk.{idx}.ffn_up_exps.weight", (cfg.n_ff, cfg.n_embd, cfg.n_expert)),
        (
            f"blk.{idx}.ffn_gate_exps.weight",
            (cfg.n_ff, cfg.n_embd, cfg.n_expert),
        ),
        (
            f"blk.{idx}.ffn_down_exps.weight",
            (cfg.n_embd, cfg.n_ff, cfg.n_expert),
        ),
    ]


# Mapping from GGUF param suffix to accessor function name.
_ACCESSOR_NAMES = {
    "attn_norm.weight": "get_attn_norm_weight",
    "ffn_norm.weight": "get_ffn_norm_weight",
    "attn_q.weight": "get_attn_q_weight",
    "attn_k.weight": "get_attn_k_weight",
    "attn_v.weight": "get_attn_v_weight",
    "attn_output.weight": "get_attn_output_weight",
    "attn_q.bias": "get_attn_q_bias",
    "attn_k.bias": "get_attn_k_bias",
    "attn_v.bias": "get_attn_v_bias",
    "attn_output.bias": "get_attn_output_bias",
    "ffn_gate_inp.weight": "get_ffn_gate_inp_weight",
    "ffn_up_exps.weight": "get_ffn_up_exps_weight",
    "ffn_gate_exps.weight": "get_ffn_gate_exps_weight",
    "ffn_down_exps.weight": "get_ffn_down_exps_weight",
}


def _shape_str(shape: tuple[int, ...]) -> str:
    """Convert shape tuple to MLIR tensor type string, e.g. '64x32xf32'."""
    return "x".join(str(d) for d in shape) + "xf32"


def _dynamic_type(ndim: int) -> str:
    """Dynamic tensor type for given rank, e.g. 'tensor<?x?xf32>'."""
    return "tensor<" + "x".join(["?"] * ndim) + "xf32>"


def _generate_accessor(
    accessor_name: str,
    gguf_name: str,
    shape: tuple[int, ...],
    scope: str = "model",
) -> str:
    """Generate a single accessor function.

    Uses flow.tensor.constant with #flow.parameter.named to load the parameter.
    Returns a dynamic-shaped tensor for compatibility with the generic layer.
    """
    static_type = f"tensor<{_shape_str(shape)}>"
    dynamic_type = _dynamic_type(len(shape))
    return f"""  util.func private @{accessor_name}(%layer: i32) -> {dynamic_type} {{
    %w = flow.tensor.constant #flow.parameter.named<"{scope}"::"{gguf_name}"> : {static_type}
    %w_dyn = tensor.cast %w : {static_type} to {dynamic_type}
    util.return %w_dyn : {dynamic_type}
  }}"""


def generate_specialized_module(cfg: MoELayerConfig, scope: str = "model") -> str:
    """Generate combined MLIR module with accessors, layer, and static wrapper.

    The module is self-contained except for compute component imports
    (rms_norm, attention_block, moe_ffn_block) which are resolved by iree-link.

    Args:
        cfg: Layer configuration with model dims and static batch/seq.
        scope: Parameter scope for flow.parameter.named (default: "model").

    Returns:
        MLIR source string for the combined module.
    """
    specs = _param_specs(cfg)
    act_shape = f"{cfg.batch}x{cfg.seq_len}x{cfg.n_embd}"
    pos_shape = f"{cfg.batch}x{cfg.seq_len}"

    # Generate accessor functions.
    accessors = []
    for gguf_name, shape in specs:
        suffix = gguf_name.split(".", 2)[2]  # Strip "blk.N." prefix
        acc_name = _ACCESSOR_NAMES[suffix]
        accessors.append(_generate_accessor(acc_name, gguf_name, shape, scope))

    accessors_str = "\n\n".join(accessors)

    # Boolean constants for use_bias and normalize_weights.
    use_bias_val = "true" if cfg.use_bias else "false"
    normalize_val = "true" if cfg.normalize_weights else "false"

    return f"""module @transformer_layer_moe_specialized {{

  // ===== Compute component imports (resolved by iree-link) =====

  util.func private @rms_norm_components.rms_norm_linalg(
      tensor<?x?xf32>, tensor<?xf32>, f32
  ) -> tensor<?x?xf32>

  util.func private @attention_block_components.attention_block(
      tensor<?x?x?xf32>, tensor<?x?xi64>,
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
      tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
      i1, index, index, index, f32, f32
  ) -> tensor<?x?x?xf32>

  util.func private @moe_ffn_components.moe_ffn_block(
      tensor<?x?xf32>, tensor<?x?xf32>,
      tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
      index, index, index, index, i1
  ) -> tensor<?x?xf32>

  // ===== Parameter accessor implementations =====

{accessors_str}

  // ===== Layer function (inlined from template) =====

  util.func private @transformer_layer_moe(
      %input: tensor<?x?x?xf32>,
      %positions: tensor<?x?xi64>,
      %layer_idx: i32,
      %n_head: index,
      %n_head_kv: index,
      %n_embd: index,
      %n_ff: index,
      %n_expert: index,
      %n_expert_used: index,
      %rms_eps: f32,
      %rope_freq_base: f32,
      %rope_freq_scale: f32,
      %use_bias: i1,
      %normalize_weights: i1
  ) -> tensor<?x?x?xf32> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %batch = tensor.dim %input, %c0 : tensor<?x?x?xf32>
    %seq_len = tensor.dim %input, %c1 : tensor<?x?x?xf32>
    %n_tokens = arith.muli %batch, %seq_len : index

    // ---- Load all parameters for this layer ----

    %attn_norm_w = util.call @get_attn_norm_weight(%layer_idx) : (i32) -> tensor<?xf32>
    %ffn_norm_w = util.call @get_ffn_norm_weight(%layer_idx) : (i32) -> tensor<?xf32>

    %wq = util.call @get_attn_q_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %wk = util.call @get_attn_k_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %wv = util.call @get_attn_v_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %wo = util.call @get_attn_output_weight(%layer_idx) : (i32) -> tensor<?x?xf32>

    %bq = util.call @get_attn_q_bias(%layer_idx) : (i32) -> tensor<?xf32>
    %bk = util.call @get_attn_k_bias(%layer_idx) : (i32) -> tensor<?xf32>
    %bv = util.call @get_attn_v_bias(%layer_idx) : (i32) -> tensor<?xf32>
    %bo = util.call @get_attn_output_bias(%layer_idx) : (i32) -> tensor<?xf32>

    %gate_inp_w = util.call @get_ffn_gate_inp_weight(%layer_idx) : (i32) -> tensor<?x?xf32>
    %up_exps_w = util.call @get_ffn_up_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf32>
    %gate_exps_w = util.call @get_ffn_gate_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf32>
    %down_exps_w = util.call @get_ffn_down_exps_weight(%layer_idx) : (i32) -> tensor<?x?x?xf32>

    // ---- Attention sub-layer ----

    %input_2d = tensor.collapse_shape %input [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>

    %attn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %input_2d, %attn_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    %attn_normed_3d = tensor.expand_shape %attn_normed [[0, 1], [2]]
        output_shape [%batch, %seq_len, %n_embd]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    %attn_out = util.call @attention_block_components.attention_block(
        %attn_normed_3d, %positions,
        %wq, %wk, %wv, %wo,
        %bq, %bk, %bv, %bo,
        %use_bias, %n_head, %n_head_kv, %n_embd,
        %rope_freq_base, %rope_freq_scale)
        : (tensor<?x?x?xf32>, tensor<?x?xi64>,
           tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
           tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
           i1, index, index, index, f32, f32) -> tensor<?x?x?xf32>

    // Residual connection: input + attn_out.
    %residual1_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf32>
    %residual1 = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    }} ins(%input, %attn_out : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%residual1_init : tensor<?x?x?xf32>) {{
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }} -> tensor<?x?x?xf32>

    // ---- MoE FFN sub-layer ----

    %residual1_2d = tensor.collapse_shape %residual1 [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>

    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(
        %residual1_2d, %ffn_norm_w, %rms_eps)
        : (tensor<?x?xf32>, tensor<?xf32>, f32) -> tensor<?x?xf32>

    %moe_out = util.call @moe_ffn_components.moe_ffn_block(
        %ffn_normed, %gate_inp_w,
        %up_exps_w, %gate_exps_w, %down_exps_w,
        %n_expert, %n_expert_used, %n_embd, %n_ff,
        %normalize_weights)
        : (tensor<?x?xf32>, tensor<?x?xf32>,
           tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
           index, index, index, index, i1) -> tensor<?x?xf32>

    %moe_out_3d = tensor.expand_shape %moe_out [[0, 1], [2]]
        output_shape [%batch, %seq_len, %n_embd]
        : tensor<?x?xf32> into tensor<?x?x?xf32>

    // Residual connection: residual1 + moe_out.
    %output_init = tensor.empty(%batch, %seq_len, %n_embd) : tensor<?x?x?xf32>
    %output = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    }} ins(%residual1, %moe_out_3d : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%output_init : tensor<?x?x?xf32>) {{
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }} -> tensor<?x?x?xf32>

    util.return %output : tensor<?x?x?xf32>
  }}

  // ===== Static-shaped entry point =====

  util.func public @forward(
      %input: tensor<{act_shape}xf32>,
      %positions: tensor<{pos_shape}xi64>
  ) -> tensor<{act_shape}xf32> {{
    %input_dyn = tensor.cast %input
        : tensor<{act_shape}xf32> to tensor<?x?x?xf32>
    %positions_dyn = tensor.cast %positions
        : tensor<{pos_shape}xi64> to tensor<?x?xi64>

    %layer_idx = arith.constant {cfg.layer_idx} : i32
    %n_head = arith.constant {cfg.n_head} : index
    %n_head_kv = arith.constant {cfg.n_head_kv} : index
    %n_embd = arith.constant {cfg.n_embd} : index
    %n_ff = arith.constant {cfg.n_ff} : index
    %n_expert = arith.constant {cfg.n_expert} : index
    %n_expert_used = arith.constant {cfg.n_expert_used} : index
    %rms_eps = arith.constant {cfg.rms_eps:e} : f32
    %rope_freq_base = arith.constant {cfg.rope_freq_base:e} : f32
    %rope_freq_scale = arith.constant {cfg.rope_freq_scale:e} : f32
    %use_bias = arith.constant {use_bias_val}
    %normalize_weights = arith.constant {normalize_val}

    %out_dyn = util.call @transformer_layer_moe(
        %input_dyn, %positions_dyn, %layer_idx,
        %n_head, %n_head_kv, %n_embd, %n_ff,
        %n_expert, %n_expert_used,
        %rms_eps, %rope_freq_base, %rope_freq_scale,
        %use_bias, %normalize_weights)
        : (tensor<?x?x?xf32>, tensor<?x?xi64>, i32,
           index, index, index, index, index, index,
           f32, f32, f32, i1, i1) -> tensor<?x?x?xf32>

    %out = tensor.cast %out_dyn
        : tensor<?x?x?xf32> to tensor<{act_shape}xf32>
    util.return %out : tensor<{act_shape}xf32>
  }}
}}"""


def generate_random_params(
    cfg: MoELayerConfig,
    seed: int = 42,
    scale: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate random parameters with GGUF naming for testing.

    Args:
        cfg: Layer configuration.
        seed: Random seed for reproducibility.
        scale: Standard deviation for random initialization.

    Returns:
        Dict mapping GGUF param names to numpy arrays (f32).
    """
    rng = np.random.RandomState(seed)
    params = {}
    for name, shape in _param_specs(cfg):
        if ".bias" in name and not cfg.use_bias:
            # Biases are zeros when not used.
            params[name] = np.zeros(shape, dtype=np.float32)
        else:
            params[name] = rng.randn(*shape).astype(np.float32) * scale
    return params
