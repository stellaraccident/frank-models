# Skill: Instantiate RMSNorm Component

## Task Description

Instantiate the RMSNorm MLIR template for a specific model configuration,
producing specialized MLIR with concrete shapes and parameters.

## Input

1. **Model configuration** containing:
   - `hidden_dim`: Hidden dimension size (e.g., 4096 for Llama-7B)
   - `dtype`: Data type (f32, f16, bf16)
   - `eps`: Epsilon value (typically 1e-5 or 1e-6)

2. **Template source**: `components/normalization/rms_norm.mlir`

## Output

Specialized MLIR module with:
- Concrete tensor shapes (hidden_dim specialized, batch dynamic)
- Model-specific function name
- Proper dtype throughout

## Process

### Step 1: Read the Template

The template at `components/normalization/rms_norm.mlir` has this signature:

```mlir
util.func public @rms_norm_linalg(
    %input: tensor<?x?xf32>,     # Dynamic batch x hidden_dim
    %weight: tensor<?xf32>,       # hidden_dim
    %eps: f32
) -> tensor<?x?xf32>
```

### Step 2: Understand the Algorithm

RMSNorm computes:
```
output = (x / sqrt(mean(x^2, axis=-1) + eps)) * weight
```

Key operations:
1. Square input elementwise
2. Reduce (sum) over hidden_dim axis
3. Divide by hidden_dim for mean
4. Add epsilon
5. Take square root
6. Divide input by RMS
7. Multiply by weight

### Step 3: Specialize for Model

For a model like Llama-7B (hidden_dim=4096):

**Change function name:**
```mlir
func.func @llama_7b_rms_norm(...)
```

**Keep dynamic batch, specialize hidden_dim:**
```mlir
%input: tensor<?x4096xf32>
%weight: tensor<4096xf32>
```

**Specialize the dim constant (if using func.func with static shapes):**
```mlir
%dim1_f32 = arith.constant 4096.0 : f32
```

Note: The template uses dynamic dimension lookup via `tensor.dim`. For specialized
versions with static hidden_dim, you can use a constant instead.

### Step 4: For Test MLIR (static shapes)

When creating test MLIR with fully static shapes:

```mlir
module @test_llama_7b_rms_norm {
  func.func @rms_norm(
      %input: tensor<1x4096xf32>,      // batch=1 for testing
      %weight: tensor<4096xf32>,
      %eps: f32
  ) -> tensor<1x4096xf32> {
    // Use static shapes throughout
    // Use arith.constant 4096.0 : f32 for dimension
    ...
  }
}
```

### Step 5: Verify

Run the specialized module through tests:
```bash
pytest tests/test_rms_norm.py -v
```

## Example Specialization

Given:
```yaml
model: llama-7b
hidden_dim: 4096
dtype: f32
eps: 1e-5
```

Produce test MLIR:
```mlir
module @llama_7b_rms_norm {
  func.func @rms_norm(
      %input: tensor<1x4096xf32>,
      %weight: tensor<4096xf32>,
      %eps: f32
  ) -> tensor<1x4096xf32> {
    // Sum of squares
    %init_sum = tensor.empty() : tensor<1xf32>
    %zero = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero : f32) outs(%init_sum : tensor<1xf32>) -> tensor<1xf32>

    %sum_sq = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : tensor<1x4096xf32>) outs(%sum_init : tensor<1xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sq = arith.mulf %in, %in : f32
      %sum = arith.addf %acc, %sq : f32
      linalg.yield %sum : f32
    } -> tensor<1xf32>

    // Compute RMS
    %dim1_f32 = arith.constant 4096.0 : f32  // Specialized!
    %rms_init = tensor.empty() : tensor<1xf32>
    %rms = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%sum_sq : tensor<1xf32>) outs(%rms_init : tensor<1xf32>) {
    ^bb0(%sum: f32, %out: f32):
      %mean = arith.divf %sum, %dim1_f32 : f32
      %with_eps = arith.addf %mean, %eps : f32
      %rms_val = math.sqrt %with_eps : f32
      linalg.yield %rms_val : f32
    } -> tensor<1xf32>

    // Normalize and scale
    %output_init = tensor.empty() : tensor<1x4096xf32>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %rms, %weight : tensor<1x4096xf32>, tensor<1xf32>, tensor<4096xf32>)
      outs(%output_init : tensor<1x4096xf32>) {
    ^bb0(%x: f32, %rms_val: f32, %w: f32, %out: f32):
      %normalized = arith.divf %x, %rms_val : f32
      %scaled = arith.mulf %normalized, %w : f32
      linalg.yield %scaled : f32
    } -> tensor<1x4096xf32>

    return %output : tensor<1x4096xf32>
  }
}
```

## Common Mistakes

1. **Forgetting to update dim constant**: The `dim1_f32` constant must match hidden_dim
2. **Wrong dtype propagation**: Ensure all operations use consistent dtype
3. **Batch dimension**: Keep batch dimension dynamic (`?`) for production, static for tests
4. **Tensor shapes mismatch**: All tensor shapes in indexing_maps must be consistent

## Validation Checklist

- [ ] Function name matches model convention
- [ ] Hidden dimension is concrete in test MLIR
- [ ] Batch dimension is appropriate (dynamic for prod, static for test)
- [ ] dim1_f32 constant matches hidden_dim
- [ ] All intermediate types are consistent
- [ ] Tests pass against numpy oracle
