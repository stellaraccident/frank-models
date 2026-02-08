"""End-to-end tests for toy MoE LLM model.

Tests the full model by linking architecture + hparams + params + components,
compiling with fake weights, and comparing against NumPy oracles.

Toy model dimensions (for fast testing):
  vocab_size=256, n_embd=64, n_head=4, n_head_kv=2, head_dim=16
  n_ff=128, n_expert=4, n_expert_used=2, n_layers=2
"""

import numpy as np
import pytest
from pathlib import Path

from tests.utils import link_and_compile_with_params, IREEConfig, LAYERS_DIR


# Paths
ARCH_DIR = Path(__file__).parent.parent.parent / "architectures"
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "llm"
COMPONENTS_DIR = Path(__file__).parent.parent.parent / "components"


# Toy model configuration
TOY_CONFIG = {
    "vocab_size": 256,
    "n_embd": 64,
    "n_head": 4,
    "n_head_kv": 2,
    "head_dim": 16,  # n_embd / n_head
    "n_ff": 128,
    "n_expert": 4,
    "n_expert_used": 2,
    "n_layers": 2,
    "rms_eps": 1e-5,
    "rope_freq_base": 10000.0,
}


def generate_toy_params(seed: int = 42) -> dict[str, np.ndarray]:
    """Generate random parameters for toy model matching GGUF naming convention.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict mapping parameter names to numpy arrays
    """
    rng = np.random.default_rng(seed)
    cfg = TOY_CONFIG
    params = {}

    # Model-level parameters
    params["token_embd.weight"] = rng.standard_normal(
        (cfg["vocab_size"], cfg["n_embd"])
    ).astype(np.float32)
    params["output_norm.weight"] = rng.standard_normal(cfg["n_embd"]).astype(np.float32)
    params["output.weight"] = rng.standard_normal(
        (cfg["n_embd"], cfg["vocab_size"])
    ).astype(np.float32)

    # Layer-level parameters
    n_embd_kv = cfg["n_head_kv"] * cfg["head_dim"]

    for layer_idx in range(cfg["n_layers"]):
        prefix = f"blk.{layer_idx}"

        # Normalization weights
        params[f"{prefix}.attn_norm.weight"] = rng.standard_normal(cfg["n_embd"]).astype(
            np.float32
        )
        params[f"{prefix}.ffn_norm.weight"] = rng.standard_normal(cfg["n_embd"]).astype(
            np.float32
        )

        # Attention projections
        params[f"{prefix}.attn_q.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.attn_k.weight"] = rng.standard_normal(
            (cfg["n_embd"], n_embd_kv)
        ).astype(np.float32)
        params[f"{prefix}.attn_v.weight"] = rng.standard_normal(
            (cfg["n_embd"], n_embd_kv)
        ).astype(np.float32)
        params[f"{prefix}.attn_output.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_embd"])
        ).astype(np.float32)

        # Attention biases (zeros since use_bias=false)
        params[f"{prefix}.attn_q.bias"] = np.zeros(cfg["n_embd"], dtype=np.float32)
        params[f"{prefix}.attn_k.bias"] = np.zeros(n_embd_kv, dtype=np.float32)
        params[f"{prefix}.attn_v.bias"] = np.zeros(n_embd_kv, dtype=np.float32)
        params[f"{prefix}.attn_output.bias"] = np.zeros(cfg["n_embd"], dtype=np.float32)

        # MoE weights
        params[f"{prefix}.ffn_gate_inp.weight"] = rng.standard_normal(
            (cfg["n_expert"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_up_exps.weight"] = rng.standard_normal(
            (cfg["n_ff"], cfg["n_embd"], cfg["n_expert"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_gate_exps.weight"] = rng.standard_normal(
            (cfg["n_ff"], cfg["n_embd"], cfg["n_expert"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_down_exps.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_ff"], cfg["n_expert"])
        ).astype(np.float32)

    return params


def _get_library_paths() -> list[str]:
    """Get all library paths needed for linking the toy model."""
    return [
        # Hparams and params
        str(MODELS_DIR / "toy" / "hparams.mlir"),
        str(MODELS_DIR / "toy" / "params.mlir"),
        # Layers
        str(LAYERS_DIR / "transformer_layer_moe_prefill.mlir"),
        str(LAYERS_DIR / "transformer_layer_moe_decode.mlir"),
        # Components
        "embedding/embedding_lookup.mlir",
        "normalization/rms_norm.mlir",
        "kvcache/kvcache.mlir",
        "attention/attention_block_prefill.mlir",
        "attention/attention_block_decode.mlir",
        "attention/attention_gqa.mlir",
        "position/rope.mlir",
        "moe/moe_ffn_block.mlir",
        "moe/mul_mat_id.mlir",
        "activation/swiglu.mlir",
    ]


class TestToyModelCompilation:
    """Test that the toy model compiles successfully."""

    def test_compiles_with_params(self, iree_cfg):
        """Test that the toy model links and compiles with parameters."""
        params = generate_toy_params(seed=42)

        model = link_and_compile_with_params(
            main_path=str(ARCH_DIR / "llm_moe_cached.mlir"),
            library_paths=_get_library_paths(),
            iree_cfg=iree_cfg,
            params=params,
            scope="model",
            debug_name="toy_llm",
        )

        # Verify the expected functions exist
        assert model.lookup_function("allocate_kv_cache") is not None
        assert model.lookup_function("prefill") is not None
        assert model.lookup_function("decode") is not None


# TODO: Add numeric validation tests once oracles are implemented
# class TestToyModelPrefill:
#     """Test prefill operation against NumPy oracle."""
#
#     def test_prefill_basic(self, iree_cfg):
#         params = generate_toy_params(seed=42)
#         model = link_and_compile_with_params(...)
#
#         # Create inputs
#         tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
#         positions = np.arange(4).reshape(1, 4).astype(np.int64)
#         ...
#
#         # Compare against oracle
#         iree_result = model.prefill(...)
#         oracle_result = prefill_oracle(...)
#         assert_close(iree_result, oracle_result)
