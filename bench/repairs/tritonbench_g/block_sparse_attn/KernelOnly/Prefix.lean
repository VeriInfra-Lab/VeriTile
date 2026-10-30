import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Arithmetic

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

/-- Actual INT32 cells, including default cells outside the allocated view,
decode to signed32 integers. Physical decoding later supplies this property. -/
def Int32Memory {α : Type} [Inhabited α] (s : State α) : Prop :=
  ∀ region ∈ (["rows","cols"] : List RegionName), ∀ offset,
    Int32Bits.inRange ((s.mem region offset).read .int)

def prefixExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (integers : Int32Memory s) :
    {t : State α // run M (prelude p) s = some t} := by
  have rows := integers "rows" (by simp)
  refine ⟨?_, ?_⟩
  swap
  simp [prelude,source,declared,ComputeKernel.surfaceBody,run,FP.Structural.step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Int32Bits.valuesValid,rows]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
