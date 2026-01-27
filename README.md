# Frank Models

MLIR component library for AI-assisted model authoring targeting IREE.

Status: Experimental. We are rolling forward this concrete implementation based on a private prototype which proofed out some of the concepts.

## Why

The goal is to produce clean IR directly from papers and reference implementations, avoiding the typical framework → converter → adapter stack that mangles programs before they reach the compiler. Generalized components are authored in MLIR with dynamic shapes and clean interfaces, validated against NumPy oracles, and composed via IREE's tooling and LLM skills and recipes.

**The DSL is the LLM skill, not a compiler dialect.**

This project originated because it was noted that modern coding agents are _very_ good at doing the kind of translations that are typically done by people and AI/ML frameworks using layer stacks of libraries, converters, deps, etc. We're going to push this and see how much of that we can outright make obsolete for production serving. This will involve adding AI skills to do a number of common transformations and compositions:

- Parameter adaptation: Specialize reference models to different parameter stores, etc.
- Quantization and numeric type specialization: Numeric transformation recipes at the IR level are easily described in language and agents are very capable of systematically applying them at model scope.
- Distribution: Recipe driven partitioning and distribution, using IREE's aysnc transfer mechanisms to overlap comms and compute as desired.
- Differentiation: Early prototypes have revealed that small amounts of sugar and LLM agents can instantiate backwards and combined forward/backward graphs from forward graphs, in a similar way that frameworks automate.
- Custom kernel specialization: Recipes to replace auto-generated blocks and kernels with custom fusions, fast path kernels, etc, is easily taught as a skill, with the IR being transformed for use/test and fed to the compiler.
- Full model-level features typically found in frameworks and bolt-on libraries: IREE has grown a full/general Tokenizer implementation (measured to be the fastest of any that could be profiled) and will be exposed to the modeling layer, allowing LLM assisted composition of full serving algorithms from base components and recipe specs. Additional services will also be added to IREE at need in order to complete the use-case scale out.

## Components

| Component                    | Purpose                             | IREE Ops Used                                   |
| ---------------------------- | ----------------------------------- | ----------------------------------------------- |
| `normalization/rms_norm`     | RMSNorm layer                       | `linalg.generic`                                |
| `activation/swiglu`          | SwiGLU activation (gate × silu(up)) | `linalg.generic`                                |
| `embedding/embedding_lookup` | Token → embedding via gather        | `iree_linalg_ext.gather`                        |
| `position/rope`              | Rotary Position Embeddings          | `linalg.generic`                                |
| `attention/attention_gqa`    | Grouped Query Attention             | `iree_linalg_ext.attention`                     |
| `attention/attention_block`  | Full attention block (QKV + RoPE)   | composition: `rope`, `attention_gqa`            |
| `moe/mul_mat_id`             | Expert-selected matrix multiply     | `iree_linalg_ext.gather`, `linalg.batch_matmul` |
| `moe/moe_ffn_block`          | Full MoE FFN with routing           | `iree_linalg_ext.topk`, `linalg.softmax`        |

## Project Structure

```
components/     # MLIR templates with dynamic shapes
oracles/        # NumPy reference implementations
tests/          # pytest comparing IREE output to oracles
skills/         # Documentation for AI-assisted tasks
```

## Running Tests

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests (requires iree-link for MoE tests)
pytest tests/ --iree-tools-dir /path/to/iree-build/tools

# Run specific component
pytest tests/test_rms_norm.py -v
```

## Component Composition

Components can call each other via `iree-link`. For example, `moe_ffn_block` declares external dependencies:

```mlir
util.func private @moe_components.mul_mat_id(...)
util.func private @activation_components.swiglu(...)
```

These are resolved at compile time by linking the component modules together.

## For AI Agents

See [CLAUDE.md](CLAUDE.md) for detailed instructions on working with this codebase, including:

- How to read and specialize MLIR templates
- MLIR → NumPy translation patterns
- Oracle generation guidelines
- Testing conventions

## Known Issues

- `attention_gqa` and `attention_block` are xfailed due to [iree-org/iree#23277](https://github.com/iree-org/iree/issues/23277) (dynamic shapes + attention op on CPU)

## License

Apache 2.0 with LLVM Exceptions (matching IREE)
