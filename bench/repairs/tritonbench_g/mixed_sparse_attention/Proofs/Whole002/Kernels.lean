import bench.repairs.tritonbench_g.mixed_sparse_attention.Kernels

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.TypedSource
open VeriTile.Triton FP.Structural
@[simp] theorem nat_tritonDiv (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) :
    Op.tritonDiv .nat bc x y=Op.floorDiv .nat bc x y := rfl
@[simp] theorem nat_tritonMod (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) :
    Op.tritonMod .nat bc x y=Op.mod .nat bc x y := rfl
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.TypedSource
