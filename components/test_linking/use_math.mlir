// Module that calls external math functions.
// Dependencies resolved by iree-link.
module @use_math {

  // External declarations - resolved by linking math_ops.mlir
  util.func private @math_ops.add_tensors(tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  util.func private @math_ops.mul_tensors(tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  // add_then_mul: (a + b) * c (elementwise)
  util.func public @add_then_mul(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>) -> tensor<?xf32> {
    %sum = util.call @math_ops.add_tensors(%a, %b) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %result = util.call @math_ops.mul_tensors(%sum, %c) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    util.return %result : tensor<?xf32>
  }

}
