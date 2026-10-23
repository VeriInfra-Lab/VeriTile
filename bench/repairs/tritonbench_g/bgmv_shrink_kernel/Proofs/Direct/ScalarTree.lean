import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.FPValues
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPScalarTree

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.ScalarTree
open VeriTile.Triton FP.Structural
noncomputable section
namespace Tree
abbrev Plans := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.FPScalarTree.Plans
abbrev eval := @VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.FPScalarTree.eval
abbrev inputs := @VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.FPScalarTree.inputs
end Tree
abbrev Plans := Tree.Plans

def reduction (M : Algebra α) (plans : Plans) : FPValues.Reduction α where
  apply := fun p dtype {shape} ax keep xs i => ofFloat dtype
    (Tree.eval (M.binary p dtype .add) (M.literal p dtype 0)
      (Tree.inputs dtype shape ax keep xs i) (plans dtype p shape ax keep i).tree)

theorem leaves (plans : Plans) (dtype : FloatDType) (h : Call) (row : Fin h.BN) :
    let plan:=plans dtype (some .fp32) [h.BN,256] ⟨1,by simp⟩ Bool.false (row,PUnit.unit)
    plan.tree.leaves.Perm ((List.finRange 256).map some++List.replicate plan.padding none) :=
  (plans dtype (some .fp32) [h.BN,256] ⟨1,by simp⟩ Bool.false (row,PUnit.unit)).valid

#axiomsClean leaves
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.ScalarTree
