// Simple tensor math operations for linking test.
module @math_ops {

  util.func public @add_tensors(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %a, %c0 : tensor<?xf32>
    %init = tensor.empty(%dim) : tensor<?xf32>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    util.return %result : tensor<?xf32>
  }

  util.func public @mul_tensors(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %a, %c0 : tensor<?xf32>
    %init = tensor.empty(%dim) : tensor<?xf32>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %prod = arith.mulf %x, %y : f32
      linalg.yield %prod : f32
    } -> tensor<?xf32>
    util.return %result : tensor<?xf32>
  }

}
