# CLAUDE.md - AI Agent Instructions for frank-models

This directory contains MLIR component templates and testing infrastructure for
AI-assisted model authoring targeting IREE compilation.

## Core Concept

**The DSL is the LLM skill, not a compiler dialect.**

You are working with:

1. **MLIR templates** in `components/` - parameterized, clean interfaces
2. **NumPy oracles** in `oracles/` - reference implementations for validation
3. **Skills** in `skills/` - documentation describing transformation tasks

Your role is to:

- Understand MLIR templates and their semantics
- Generate specialized variants from templates
- Create numpy oracles that match MLIR semantics exactly
- Ensure numerical correctness through testing

## Directory Layout

```
components/     # MLIR templates (READ, UNDERSTAND, SPECIALIZE)
oracles/        # NumPy reference implementations (WRITE, MAINTAIN)
tests/          # pytest tests comparing IREE output to oracles
skills/         # Task documentation (READ, FOLLOW)
```

## Working with Components

### Reading MLIR Templates

Components in `components/` use these patterns:

1. **Dynamic shapes**: `tensor<?x?xf32>` - shapes determined at runtime
2. **util.func**: IREE-specific function declaration with streaming support
3. **linalg.generic**: Core computational pattern with affine maps
4. **flow.dispatch.region**: Explicit fusion boundary (when present)

Example signature to understand:

```mlir
util.func public @rms_norm_linalg(
    %input: tensor<?x?xf32>,      // [batch, hidden_dim]
    %weight: tensor<?xf32>,        // [hidden_dim]
    %eps: f32                      // epsilon for numerical stability
) -> tensor<?x?xf32>
```

### Instantiation Tasks

When asked to instantiate a component for a specific model:

1. **Read the template** in `components/`
2. **Read the skill doc** in `skills/instantiation/`
3. **Determine concrete shapes** from model config
4. **Produce specialized MLIR** with concrete types
5. **Verify** by running tests

### Oracle Generation

When asked to create a numpy oracle from MLIR:

1. **Read the MLIR template** carefully
2. **Map linalg.generic operations** to numpy equivalents
3. **Match semantics exactly**:
   - Same reduction axes
   - Same epsilon handling
   - Same dtype behavior
4. **Write the oracle** to `oracles/`
5. **Add tests** that validate against the oracle

## MLIR to NumPy Translation Patterns

| MLIR Pattern                                           | NumPy Equivalent            |
| ------------------------------------------------------ | --------------------------- |
| `linalg.fill` with reduction                           | `np.zeros()` initialization |
| `arith.mulf` in generic                                | `*` (elementwise)           |
| `arith.addf` in generic                                | `+` (elementwise)           |
| `arith.divf` in generic                                | `/` (elementwise)           |
| `math.sqrt`                                            | `np.sqrt()`                 |
| `math.exp`                                             | `np.exp()`                  |
| Reduction `iterator_types = ["parallel", "reduction"]` | `axis=-1` or explicit axis  |
| `affine_map<(d0, d1) -> (d0)>` for output              | Reduction over d1           |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific component
pytest tests/test_rms_norm.py -v

# Run with CPU backend
pytest tests/ --backend=llvm-cpu

# Run with GPU backend (when available)
pytest tests/ --backend=rocm
```

## Before Committing

Run pre-commit to format code and markdown:

```bash
pre-commit run --all-files
```

This runs:

- **black** - Python formatting
- **prettier** - Markdown and YAML formatting

## Key Constraints

1. **No new dialects** - Use existing IREE infrastructure
2. **Clean interfaces** - Function signatures are the API
3. **Numerical correctness** - Must match reference within tolerance
4. **Composition** - Components should compose cleanly

## When Uncertain

1. Check `skills/` for task-specific guidance
2. Look at existing `oracles/` for patterns
3. Run tests to validate understanding
4. Ask for clarification rather than guess

## File Conventions

- MLIR files: lowercase_with_underscores.mlir
- Python files: lowercase_with_underscores.py
- Test files: test\_<component>.py
- Skill docs: lowercase.md

## Error Tolerance

For numerical comparison:

- f32: `rtol=1e-5, atol=1e-6`
- f16: `rtol=1e-3, atol=1e-4`
- Quantized: depends on format, documented per-test
