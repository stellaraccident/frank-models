# Skill: Generate NumPy Oracle from MLIR

## Task Description

Given an MLIR component, generate a matching NumPy reference implementation
that can be used as an oracle for testing.

## Input

1. **MLIR source file** from `components/`
2. **Understanding of the mathematical operation** it implements

## Output

1. **Python function** in `oracles/` that computes the same result
2. **Docstring** explaining the correspondence to MLIR

## Process

### Step 1: Identify Function Signature

From MLIR like:

```mlir
util.func public @rms_norm_linalg(
    %input: tensor<?x?xf32>,
    %weight: tensor<?xf32>,
    %eps: f32
) -> tensor<?x?xf32>
```

Create Python signature:

```python
def rms_norm(
    x: np.ndarray,        # corresponds to %input
    weight: np.ndarray,   # corresponds to %weight
    eps: float = 1e-6,    # corresponds to %eps
) -> np.ndarray:
```

### Step 2: Map linalg.generic to NumPy

For each `linalg.generic`:

1. **Identify indexing maps** - these define the iteration space
2. **Identify iterator types** - "parallel" or "reduction"
3. **Translate the body** - map arith/math ops to numpy

#### Example: Sum of Squares Reduction

MLIR:

```mlir
%sum_sq = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,  # input: 2D
    affine_map<(d0, d1) -> (d0)>        # output: 1D (reduce d1)
  ],
  iterator_types = ["parallel", "reduction"]
} ins(%input : tensor<?x?xf32>) outs(%sum_init : tensor<?xf32>) {
^bb0(%in: f32, %acc: f32):
  %sq = arith.mulf %in, %in : f32
  %sum = arith.addf %acc, %sq : f32
  linalg.yield %sum : f32
}
```

Analysis:

- Input shape: `(d0, d1)` - 2D
- Output shape: `(d0)` - 1D, so d1 is reduced
- Body: square then accumulate
- iterator_types: d1 is "reduction"

NumPy:

```python
# d1 (axis=-1) is reduced
sum_sq = np.sum(x * x, axis=-1, keepdims=True)
```

### Step 3: Handle Broadcasting

MLIR indexing maps show exactly how tensors broadcast.

`affine_map<(d0, d1) -> (d1)>` means:

- Input is 1D with size d1
- It broadcasts across d0

In NumPy, this is handled automatically if shapes are compatible,
but you may need to ensure the weight has the right shape.

### Step 4: Preserve Precision Semantics

- Use `np.float32` by default to match f32
- For f16, use `np.float16` (with care for accumulation)
- Keep intermediate precision consistent with MLIR

### Step 5: Document the Correspondence

Add docstrings that reference the MLIR:

```python
def rms_norm(x, weight, eps=1e-6):
    """RMS Normalization.

    This matches the semantics of @rms_norm_linalg in
    components/normalization/rms_norm.mlir

    Algorithm (matching MLIR):
        1. Compute sum of squares: sum_sq = sum(x^2, axis=-1)
        2. Compute mean: mean_sq = sum_sq / hidden_dim
        3. Compute RMS: rms = sqrt(mean_sq + eps)
        4. Normalize: normalized = x / rms
        5. Scale: output = normalized * weight
    """
```

## Translation Table

| MLIR                      | NumPy                       |
| ------------------------- | --------------------------- |
| `arith.mulf`              | `*`                         |
| `arith.addf`              | `+`                         |
| `arith.subf`              | `-`                         |
| `arith.divf`              | `/`                         |
| `arith.negf`              | `-x` (unary)                |
| `math.sqrt`               | `np.sqrt`                   |
| `math.exp`                | `np.exp`                    |
| `math.log`                | `np.log`                    |
| `math.cos`                | `np.cos`                    |
| `math.sin`                | `np.sin`                    |
| `math.tanh`               | `np.tanh`                   |
| `arith.maximumf`          | `np.maximum`                |
| `arith.minimumf`          | `np.minimum`                |
| `linalg.fill` + reduction | `np.zeros` or explicit init |

## Common Patterns

### Softmax

```mlir
// MLIR: exp(x - max(x)) / sum(exp(x - max(x)))
```

```python
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### SiLU (Swish)

```mlir
// MLIR: x * sigmoid(x) where sigmoid = 1 / (1 + exp(-x))
```

```python
def silu(x):
    return x / (1 + np.exp(-x))
```

Or equivalently:

```python
def silu(x):
    return x * (1 / (1 + np.exp(-x)))
```

### RoPE (Rotary Position Embeddings)

Complex - involves cos/sin applied to dimension pairs. See the
MLIR in `components/positional/rope.mlir` for the full pattern.

## Understanding Reduction Axes

The key to translating `linalg.generic` is understanding which dimensions are reduced:

1. Look at `iterator_types`: `"reduction"` means that dimension is summed/accumulated
2. Look at output indexing map: dimensions missing from output are reduced
3. Map to numpy `axis` parameter

Example:

```mlir
iterator_types = ["parallel", "parallel", "reduction"]
# d0, d1 are parallel (preserved), d2 is reduced
# In numpy: axis=-1 or axis=2
```

## Validation

After writing oracle:

1. Write test that compiles MLIR and runs both
2. Use `np.testing.assert_allclose` with appropriate tolerance
3. Test edge cases (zeros, large values, NaN handling)

## Example: Complete Oracle

```python
def rms_norm(x, weight, eps=1e-6):
    """RMS Normalization matching @rms_norm_linalg."""
    # Step 1-2: Sum of squares and mean
    sum_sq = np.sum(x * x, axis=-1, keepdims=True)
    hidden_dim = x.shape[-1]
    mean_sq = sum_sq / hidden_dim

    # Step 3: RMS
    rms = np.sqrt(mean_sq + eps)

    # Step 4-5: Normalize and scale
    normalized = x / rms
    return normalized * weight
```
