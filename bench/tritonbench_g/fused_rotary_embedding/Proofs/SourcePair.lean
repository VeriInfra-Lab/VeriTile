import bench.tritonbench_g.fused_rotary_embedding.Proofs.PublicIO
import bench.tritonbench_g.fused_rotary_embedding.Proofs.IOInterpretation
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.SourcePair
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def source (shared : Bool) (p : Source.Parameters) :=
  if shared then Source.kernelSharedQOffset p else Source.kernel p
/-- Proof-only initialBlock through both Q-address assignments. -/
def initialBlock (shared : Bool) (p : Source.Parameters) := (source shared p).surfaceBody.take 9

theorem split (shared : Bool) (p : Source.Parameters) :
    (source shared p).surfaceBody = initialBlock shared p ++ (Source.kernel p).surfaceBody.drop 9 := by
  cases shared <;> rfl

theorem initialBlock_exec {α : Type} [Inhabited α] (A : Algebra α) (p : Source.Parameters) (s : State α) :
    run A (initialBlock Bool.false p) s = run A (initialBlock Bool.true p) s := by
  have widthNonneg : ¬ (p.HEAD_DIM : Int) < 0 := by omega
  simp [initialBlock,source,Source.kernel,Source.kernelSharedQOffset,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,State.setReg,numeric,numericLt,natLt,numericEq,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv,IntegerExpr.sameShape,widthNonneg]
  funext dtype shape name
  by_cases hn : name = "off_q1"
  · subst name
    by_cases hd : TileDType.int = dtype
    · subst dtype
      by_cases hs : [p.BLOCK_HALF] = shape
      · subst shape
        simp only [↓reduceIte,↓reduceDIte,Option.some.injEq]
        funext i
        simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]
        ring
      · simp [hs]
    · simp [hd]
  · simp only [if_neg hn]

theorem initialBlock_safe {α : Type} [Inhabited α] (A : Algebra α) (p : Source.Parameters)
    (shared : Bool) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList A bounds (initialBlock shared p) s := by
  cases shared <;>
    simp [initialBlock,source,Source.kernel,Source.kernelSharedQOffset,ComputeKernel.surfaceBody,
      Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeOp_unfold]

theorem source_exec {α : Type} [Inhabited α] (A : Algebra α) (p : Source.Parameters) (s : State α) :
    exec A (Source.kernel p) s = exec A (Source.kernelSharedQOffset p) s := by
  change run A (source Bool.false p).surfaceBody s = run A (source Bool.true p).surfaceBody s
  simp only [split,run_append,initialBlock_exec]

theorem source_safe {α : Type} [Inhabited α] (A : Algebra α) (p : Source.Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    TraceSafe A bounds (Source.kernel p) s ↔ TraceSafe A bounds (Source.kernelSharedQOffset p) s := by
  change Safety.traceSafeList A bounds (source Bool.false p).surfaceBody s ↔
    Safety.traceSafeList A bounds (source Bool.true p).surfaceBody s
  simp only [split,Safety.traceSafeList_append,run_append,initialBlock_exec,initialBlock_safe]
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.SourcePair
