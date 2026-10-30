import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.Engine
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def entry (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    {t : State α // run (engine (withDot M dot)) (parts p).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]
  rfl

@[simp] theorem entry_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    (entry M dot p s).val.mem=s.mem := by simp [entry]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Source
