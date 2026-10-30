import bench.repairs.tritonbench_g.attention_kernel.Proofs.Engine
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def entryAs (M : Algebra α) (dot : Dot α) (storage : TypedSource.Storage) (p : Parameters) (s : State α) :
    {t : State α // run (engine (withDot M dot)) (partsAs storage p).before s=some t} := by
  cases storage
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [partsAs,kernelAs,TypedSource.repaired,TypedSource.FP16.repaired,TypedSource.BF16.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]
    rfl

def entry (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :=
  entryAs M dot p.storage p s

@[simp] theorem entry_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    (entry M dot p s).val.mem=s.mem := by
  unfold entry
  generalize p.storage=storage
  cases storage <;> simp [entryAs]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
