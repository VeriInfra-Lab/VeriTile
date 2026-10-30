import bench.repairs.tritonbench_g.streamk_matmul.Whole001.SourceDefinitions

namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Reference
open Source Protocol Plan
noncomputable section

/-- Independent Real target, distinct from all launch, chunk and native graphs. -/
def matrixProduct (p : Parameters) (a : A p → ℝ) (b : B p → ℝ)
    (row : Fin p.h.m) (col : Fin p.h.n) : ℝ :=
  ∑ k : Fin p.h.k,
    a ⟨row.val*p.strideAM+k.val*p.strideAK,Geometry.input_mask_safe _ _ _ _ _ _ p.inputA row.isLt k.isLt⟩ *
    b ⟨k.val*p.strideBK+col.val*p.strideBN,Geometry.input_mask_safe _ _ _ _ _ _ p.inputB k.isLt col.isLt⟩

end
end Reference
end VeriTile.Bench.Repairs.StreamK.Whole001
