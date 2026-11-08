import VeriTile.Triton.Float.StructuralSafety
import bench.tritonbench_g.attention_forward_triton.Proofs.Engine
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
  simp [parts,kernel,TypedSource.repaired,originalKernel,
    ComputeKernel.surfaceBody,counter,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]
  rfl

@[simp] theorem entry_memory (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    (entry M dot p s).val.mem=s.mem := by simp [entry]

/-- The actual helper's local lo/hi bounds evaluate once to the original
runtime interval. This connects the source dynamic node to the range proof. -/
theorem entry_loop (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    step (engine (withDot M dot))
      (.forRangeDyn counter (parts p).start (parts p).stop (parts p).stride (parts p).body)
      (entry M dot p s).val =
    range (engine (withDot M dot)) counter 0 p.length 64 (parts p).body
      (entry M dot p s).val := by
  simp [parts,kernel,TypedSource.repaired,originalKernel,entry,
    ComputeKernel.surfaceBody,counter,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]

theorem entry_loop_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.stepSafe (engine (withDot M dot)) bounds
      (.forRangeDyn counter (parts p).start (parts p).stop (parts p).stride (parts p).body)
      (entry M dot p s).val =
    Safety.rangeSafe (engine (withDot M dot)) bounds counter 0 p.length 64 (parts p).body
      (entry M dot p s).val := by
  simp [Safety.stepSafe_unfold,Safety.safeOp_unfold,parts,kernel,TypedSource.repaired,originalKernel,entry,
    ComputeKernel.surfaceBody,counter,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]

#axiomsClean entry_loop_safe

#axiomsClean entry_loop

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Source
