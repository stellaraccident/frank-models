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


# Mapping from GGUF param suffix to accessor function name in model_params namespace.
# The namespace indicates these are parameters, so we drop the "get_" prefix.
_ACCESSOR_NAMES = {
    "attn_norm.weight": "attn_norm_weight",
    "ffn_norm.weight": "ffn_norm_weight",
    "attn_q.weight": "attn_q_weight",
    "attn_k.weight": "attn_k_weight",
    "attn_v.weight": "attn_v_weight",
    "attn_output.weight": "attn_output_weight",
    "attn_q.bias": "attn_q_bias",
    "attn_k.bias": "attn_k_bias",
    "attn_v.bias": "attn_v_bias",
    "attn_output.bias": "attn_output_bias",
    "ffn_gate_inp.weight": "ffn_gate_inp_weight",
    "ffn_up_exps.weight": "ffn_up_exps_weight",
    "ffn_gate_exps.weight": "ffn_gate_exps_weight",
    "ffn_down_exps.weight": "ffn_down_exps_weight",
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
    return f"""  util.func public @{accessor_name}(%layer: i32) -> {dynamic_type} {{
    %w = flow.tensor.constant #flow.parameter.named<"{scope}"::"{gguf_name}"> : {static_type}
    %w_dyn = tensor.cast %w : {static_type} to {dynamic_type}
    util.return %w_dyn : {dynamic_type}
  }}"""


def generate_accessor_module(cfg: MoELayerConfig, scope: str = "model") -> str:
    """Generate MLIR module with parameter accessor implementations.

    Args:
        cfg: Layer configuration.
        scope: Parameter scope for flow.parameter.named (default: "model").

    Returns:
        MLIR source string for the accessor module.
    """
    specs = _param_specs(cfg)
    accessors = []
    for gguf_name, shape in specs:
        suffix = gguf_name.split(".", 2)[2]  # Strip "blk.N." prefix
        acc_name = _ACCESSOR_NAMES[suffix]
        accessors.append(_generate_accessor(acc_name, gguf_name, shape, scope))

    accessors_str = "\n\n".join(accessors)

    # Accessors in model_params namespace for dependency injection pattern
    return f"""module @model_params {{

{accessors_str}

}}"""


def generate_wrapper_module(cfg: MoELayerConfig) -> str:
    """Generate main module with static @forward wrapper that calls the layer.

    Args:
        cfg: Layer configuration with model dims and static batch/seq.

    Returns:
        MLIR source string for the main wrapper module.
    """
    act_shape = f"{cfg.batch}x{cfg.seq_len}x{cfg.n_embd}"
    pos_shape = f"{cfg.batch}x{cfg.seq_len}"
    use_bias_val = "true" if cfg.use_bias else "false"
    normalize_val = "true" if cfg.normalize_weights else "false"

    return f"""module @main {{

  // Import layer function (resolved by iree-link from layers/transformer_layer_moe.mlir)
  util.func private @transformer_layer_moe_components.transformer_layer_moe(
      tensor<?x?x?xf32>, tensor<?x?xi64>, i32,
      index, index, index, index, index, index,
      f32, f32, f32, i1, i1
  ) -> tensor<?x?x?xf32>

  // Static-shaped entry point

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

    %out_dyn = util.call @transformer_layer_moe_components.transformer_layer_moe(
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
