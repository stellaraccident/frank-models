"""Compilation tests for LLM model forward passes.

These tests verify that the model MLIR files compile successfully.
Numerical validation will be added in a future step with proper integration testing.
"""

import pytest
import tempfile
from pathlib import Path

from tests.utils import link_and_compile


def _compile_model(iree_cfg, model_name: str):
    """Compile a model MLIR file with minimal accessors/hparams for smoke test.

    This generates stub hparams and model_params modules to satisfy imports,
    then links with the model and attempts compilation.

    Args:
        iree_cfg: Pytest fixture with IREE tool paths
        model_name: Name of the model (mixtral, grok, qwen2moe, dbrx)

    Returns:
        True if compilation succeeds (we don't return the module since we're not running it)
    """
    model_mlir_path = (
        Path(__file__).parent.parent.parent / "models" / "llm" / f"{model_name}.mlir"
    )

    # Generate stub hparams module that returns dummy values
    hparams_mlir = """
module @hparams {
  util.func public @vocab_size() -> i64 {
    %v = arith.constant 32000 : i64
    util.return %v : i64
  }
  util.func public @block_count() -> i64 {
    %v = arith.constant 2 : i64
    util.return %v : i64
  }
  util.func public @embedding_length() -> i64 {
    %v = arith.constant 2048 : i64
    util.return %v : i64
  }
  util.func public @attention_head_count() -> i64 {
    %v = arith.constant 16 : i64
    util.return %v : i64
  }
  util.func public @attention_head_count_kv() -> i64 {
    %v = arith.constant 4 : i64
    util.return %v : i64
  }
  util.func public @feed_forward_length() -> i64 {
    %v = arith.constant 8192 : i64
    util.return %v : i64
  }
  util.func public @expert_count() -> i64 {
    %v = arith.constant 4 : i64
    util.return %v : i64
  }
  util.func public @expert_used_count() -> i64 {
    %v = arith.constant 2 : i64
    util.return %v : i64
  }
  util.func public @rope_freq_base() -> f32 {
    %v = arith.constant 10000.0 : f32
    util.return %v : f32
  }
  util.func public @layer_norm_rms_epsilon() -> f32 {
    %v = arith.constant 1.0e-5 : f32
    util.return %v : f32
  }
}
"""

    # Generate stub model_params module that returns dummy tensors
    # For model-level params (token_embd, output_norm, output)
    model_params_mlir = """
module @model_params {
  util.func public @token_embd_weight() -> tensor<?x?xf32> {
    %t = util.unfoldable_constant dense<0.0> : tensor<32000x2048xf32>
    %d = tensor.cast %t : tensor<32000x2048xf32> to tensor<?x?xf32>
    util.return %d : tensor<?x?xf32>
  }
  util.func public @output_norm_weight() -> tensor<?xf32> {
    %t = util.unfoldable_constant dense<1.0> : tensor<2048xf32>
    %d = tensor.cast %t : tensor<2048xf32> to tensor<?xf32>
    util.return %d : tensor<?xf32>
  }
  util.func public @output_weight() -> tensor<?x?xf32> {
    %t = util.unfoldable_constant dense<0.0> : tensor<2048x32000xf32>
    %d = tensor.cast %t : tensor<2048x32000xf32> to tensor<?x?xf32>
    util.return %d : tensor<?x?xf32>
  }

  // Layer-level parameter accessors (for transformer_layer_moe)
  util.func public @attn_norm_weight(%layer: i32) -> tensor<?xf32> {
    %t = util.unfoldable_constant dense<1.0> : tensor<2048xf32>
    %d = tensor.cast %t : tensor<2048xf32> to tensor<?xf32>
    util.return %d : tensor<?xf32>
  }
  util.func public @ffn_norm_weight(%layer: i32) -> tensor<?xf32> {
    %t = util.unfoldable_constant dense<1.0> : tensor<2048xf32>
    %d = tensor.cast %t : tensor<2048xf32> to tensor<?xf32>
    util.return %d : tensor<?xf32>
  }
  util.func public @attn_q_weight(%layer: i32) -> tensor<?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<2048x2048xf32>
    %d = tensor.cast %t : tensor<2048x2048xf32> to tensor<?x?xf32>
    util.return %d : tensor<?x?xf32>
  }
  util.func public @attn_k_weight(%layer: i32) -> tensor<?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<2048x512xf32>
    %d = tensor.cast %t : tensor<2048x512xf32> to tensor<?x?xf32>
    util.return %d : tensor<?x?xf32>
  }
  util.func public @attn_v_weight(%layer: i32) -> tensor<?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<2048x512xf32>
    %d = tensor.cast %t : tensor<2048x512xf32> to tensor<?x?xf32>
    util.return %d : tensor<?x?xf32>
  }
  util.func public @attn_output_weight(%layer: i32) -> tensor<?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<2048x2048xf32>
    %d = tensor.cast %t : tensor<2048x2048xf32> to tensor<?x?xf32>
    util.return %d : tensor<?x?xf32>
  }
  util.func public @attn_q_bias(%layer: i32) -> tensor<?xf32> {
    %t = util.unfoldable_constant dense<0.0> : tensor<2048xf32>
    %d = tensor.cast %t : tensor<2048xf32> to tensor<?xf32>
    util.return %d : tensor<?xf32>
  }
  util.func public @attn_k_bias(%layer: i32) -> tensor<?xf32> {
    %t = util.unfoldable_constant dense<0.0> : tensor<512xf32>
    %d = tensor.cast %t : tensor<512xf32> to tensor<?xf32>
    util.return %d : tensor<?xf32>
  }
  util.func public @attn_v_bias(%layer: i32) -> tensor<?xf32> {
    %t = util.unfoldable_constant dense<0.0> : tensor<512xf32>
    %d = tensor.cast %t : tensor<512xf32> to tensor<?xf32>
    util.return %d : tensor<?xf32>
  }
  util.func public @attn_output_bias(%layer: i32) -> tensor<?xf32> {
    %t = util.unfoldable_constant dense<0.0> : tensor<2048xf32>
    %d = tensor.cast %t : tensor<2048xf32> to tensor<?xf32>
    util.return %d : tensor<?xf32>
  }
  util.func public @ffn_gate_inp_weight(%layer: i32) -> tensor<?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<4x2048xf32>
    %d = tensor.cast %t : tensor<4x2048xf32> to tensor<?x?xf32>
    util.return %d : tensor<?x?xf32>
  }
  util.func public @ffn_up_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<8192x2048x4xf32>
    %d = tensor.cast %t : tensor<8192x2048x4xf32> to tensor<?x?x?xf32>
    util.return %d : tensor<?x?x?xf32>
  }
  util.func public @ffn_gate_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<8192x2048x4xf32>
    %d = tensor.cast %t : tensor<8192x2048x4xf32> to tensor<?x?x?xf32>
    util.return %d : tensor<?x?x?xf32>
  }
  util.func public @ffn_down_exps_weight(%layer: i32) -> tensor<?x?x?xf32> {
    %t = util.unfoldable_constant dense<0.1> : tensor<2048x8192x4xf32>
    %d = tensor.cast %t : tensor<2048x8192x4xf32> to tensor<?x?x?xf32>
    util.return %d : tensor<?x?x?xf32>
  }
}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as hf:
        hf.write(hparams_mlir)
        hparams_path = hf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as mf:
        mf.write(model_params_mlir)
        model_params_path = mf.name

    try:
        # Build component and layer library lists
        layers_dir = Path(__file__).parent.parent.parent / "layers"

        # Absolute paths for hparams, model_params, and layer (libraries)
        abs_libraries = [
            hparams_path,
            model_params_path,
            str(layers_dir / "transformer_layer_moe.mlir"),
        ]

        # Component paths (relative to components/)
        component_libraries = [
            "embedding/embedding_lookup.mlir",
            "normalization/rms_norm.mlir",
            "attention/attention_block.mlir",
            "attention/attention_gqa.mlir",
            "position/rope.mlir",
            "moe/moe_ffn_block.mlir",
            "moe/mul_mat_id.mlir",
            "activation/swiglu.mlir",
        ]

        # Compile using existing utility (this handles linking and compilation)
        # We just need it to compile successfully, don't need to run it
        link_and_compile(
            main_path=str(model_mlir_path),
            library_paths=abs_libraries + component_libraries,
            iree_cfg=iree_cfg,
            debug_name=f"{model_name}_compile_test",
        )

        return True

    finally:
        Path(hparams_path).unlink(missing_ok=True)
        Path(model_params_path).unlink(missing_ok=True)


@pytest.mark.slow
@pytest.mark.xfail(
    reason="expand_shape dimension inference fails through moe_ffn_block call - PropagateLinalgTransposePass can't trace n_tokens relationship"
)
def test_mixtral_compiles(iree_cfg):
    """Test that Mixtral model MLIR compiles."""
    assert _compile_model(iree_cfg, "mixtral")


@pytest.mark.slow
@pytest.mark.xfail(
    reason="expand_shape dimension inference fails through moe_ffn_block call - PropagateLinalgTransposePass can't trace n_tokens relationship"
)
def test_grok_compiles(iree_cfg):
    """Test that Grok model MLIR compiles."""
    assert _compile_model(iree_cfg, "grok")


@pytest.mark.slow
@pytest.mark.xfail(
    reason="expand_shape dimension inference fails through moe_ffn_block call - PropagateLinalgTransposePass can't trace n_tokens relationship"
)
def test_qwen2moe_compiles(iree_cfg):
    """Test that Qwen2MoE model MLIR compiles."""
    assert _compile_model(iree_cfg, "qwen2moe")


@pytest.mark.slow
@pytest.mark.xfail(
    reason="expand_shape dimension inference fails through moe_ffn_block call - PropagateLinalgTransposePass can't trace n_tokens relationship"
)
def test_dbrx_compiles(iree_cfg):
    """Test that DBRX model MLIR compiles."""
    assert _compile_model(iree_cfg, "dbrx")
