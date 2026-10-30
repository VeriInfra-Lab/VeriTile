import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IOResult
import bench.repairs.tritonbench_g.fused_recurrent_delta.RuntimeIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.SourcePair
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- Proof-only first eight statements: PID/address setup through p_o. -/
def initialBlock (scheduled : Bool) (s : Shape) (f : Flags) :=
  (RuntimeIO.source .forward scheduled s f).surfaceBody.take 8
 theorem split (scheduled : Bool) (s : Shape) (f : Flags) :
    (RuntimeIO.source .forward scheduled s f).surfaceBody=initialBlock scheduled s f++
      (RuntimeIO.source .forward Bool.false s f).surfaceBody.drop 8 := by
  cases scheduled <;> cases hb : f.vectorBeta <;>
    simp only [RuntimeIO.source,Runtime.forward,Runtime.forwardDirectOutput,initialBlock,
      hb,Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody] <;> rfl

 theorem block_exec (M : Algebra α) (s : Shape) (f : Flags) (u : State α)
    (singleKey : u.pids 1=0) :
    run M (initialBlock Bool.false s f) u=run M (initialBlock Bool.true s f) u := by
  cases hb : f.vectorBeta <;>
    simp [initialBlock,RuntimeIO.source,Runtime.forward,Runtime.forwardDirectOutput,
      hb,ComputeKernel.surfaceBody,run,step,evalExpr,FP.Structural.evalOp_unfold,
      State.setReg,singleKey,numeric,bop]
  all_goals
    funext dtype shape name
    by_cases hn : name="p_o"
    · subst name
      by_cases hd : TileDType.ptr=dtype
      · subst dtype
        by_cases hs : [s.FBV]=shape
        · subst shape
          simp only [↓reduceIte,↓reduceDIte,Option.some.injEq]
          funext i
          simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
        · simp [hs]
      · simp [hd]
    · simp only [if_neg hn]

 theorem block_safe (M : Algebra α) (s : Shape) (f : Flags) (scheduled : Bool)
    (bounds : RegionName → Nat) (u : State α) :
    Safety.traceSafeList M bounds (initialBlock scheduled s f) u := by
  cases scheduled <;> cases hb : f.vectorBeta <;>
    simp [initialBlock,RuntimeIO.source,Runtime.forward,Runtime.forwardDirectOutput,
      hb,ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.stepSafe,
      Safety.safeExpr,Safety.safeOp_unfold]

 theorem executes (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags) (u : State α)
    (singleKey : u.pids 1=0) :
    exec M (RuntimeIO.source kind Bool.false s f) u=exec M (RuntimeIO.source kind Bool.true s f) u := by
  cases kind with
  | backward => rfl
  | forward =>
    change run M (RuntimeIO.source .forward Bool.false s f).surfaceBody u=
      run M (RuntimeIO.source .forward Bool.true s f).surfaceBody u
    conv_lhs => rw [split Bool.false s f]
    conv_rhs => rw [split Bool.true s f]
    simp only [run_append]
    rw [block_exec M s f u singleKey]

 theorem safe (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags) (u : State α)
    (singleKey : u.pids 1=0) (bounds : RegionName → Nat) :
    TraceSafe M bounds (RuntimeIO.source kind Bool.false s f) u ↔
      TraceSafe M bounds (RuntimeIO.source kind Bool.true s f) u := by
  cases kind with
  | backward => rfl
  | forward =>
    change Safety.traceSafeList M bounds (RuntimeIO.source .forward Bool.false s f).surfaceBody u ↔
      Safety.traceSafeList M bounds (RuntimeIO.source .forward Bool.true s f).surfaceBody u
    conv_lhs => rw [split Bool.false s f]
    conv_rhs => rw [split Bool.true s f]
    simp only [Safety.traceSafeList_append,block_safe,true_and]
    rw [block_exec M s f u singleKey]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.SourcePair
