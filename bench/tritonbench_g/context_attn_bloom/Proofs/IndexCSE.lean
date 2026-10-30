import bench.tritonbench_g.context_attn_bloom.Proofs.ReadableSource
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IndexCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

abbrev original (p : Math.Layout) (r : Program.Regions) (scale : ℝ) :=
  ContextAttnBloomSource.atLayout ContextAttnBloomSource.kernel p r scale
abbrev candidate (p : Math.Layout) (r : Program.Regions) (scale : ℝ) :=
  ContextAttnBloomSource.atLayout ContextAttnBloomSource.indexShared p r scale

/-- Proof-only split after row-lane generation. The complete remaining
floating computation, key loop, metadata use and stores are unchanged. -/
def sourcePrefix (p : Math.Layout) (r : Program.Regions) (scale : ℝ) :=
  (original p r scale).surfaceBody.take 12
def candidatePrefix (p : Math.Layout) (r : Program.Regions) (scale : ℝ) :=
  (candidate p r scale).surfaceBody.take 12
def suffix (p : Math.Layout) (r : Program.Regions) (scale : ℝ) :=
  (original p r scale).surfaceBody.drop 12

theorem candidate_body (p : Math.Layout) (r : Program.Regions) (scale : ℝ) :
    (candidate p r scale).surfaceBody=candidatePrefix p r scale ++ suffix p r scale := rfl

theorem prefix_execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Math.Layout) (r : Program.Regions) (scale : ℝ) (s : State α) :
    run M (sourcePrefix p r scale) s=run M (candidatePrefix p r scale) s := by
  simp [sourcePrefix,candidatePrefix,original,candidate,ContextAttnBloomSource.atLayout,
    ContextAttnBloomSource.kernel,ContextAttnBloomSource.indexShared,ComputeKernel.surfaceBody,List.take,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,integralFloorDiv,integralMod,bop,
    ofFloat,toFloat,Region.cast,Option.bind,State.setReg,fold_writes_regs,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Function.comp_def,Nat.mul_comm]
  have product : FP.Structural.bop (fun (x y : Nat)=>x*y) (.nil : Broadcast [] [] [])
      (fun _=>s.pids 2) (fun _=>p.BM)=
      FP.Structural.bop (fun (x y : Nat)=>x*y) (.nil : Broadcast [] [] [])
        (fun _=>p.BM) (fun _=>s.pids 2) := by
    funext i
    exact Nat.mul_comm _ _
  simp only [product]

theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (p : Math.Layout) (r : Program.Regions) (scale : ℝ) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (sourcePrefix p r scale) s ↔
      Safety.traceSafeList M bounds (candidatePrefix p r scale) s := by
  simp (config := {maxSteps := 200000}) [sourcePrefix,candidatePrefix,original,candidate,
    ContextAttnBloomSource.atLayout,ContextAttnBloomSource.kernel,ContextAttnBloomSource.indexShared,
    ComputeKernel.surfaceBody,List.take,Safety.traceSafeList_cons,Safety.traceSafeList_nil,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,integralFloorDiv,integralMod,bop,
    ofFloat,toFloat,Region.cast,Option.bind,State.setReg,fold_writes_regs,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Function.comp_def,Nat.mul_comm]

theorem execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Math.Layout) (r : Program.Regions) (scale : ℝ) (s : State α) :
    exec M (original p r scale) s=exec M (candidate p r scale) s := by
  change run M (original p r scale).surfaceBody s=run M (candidate p r scale).surfaceBody s
  rw [candidate_body]
  conv_lhs=>rw [←List.take_append_drop 12 (original p r scale).surfaceBody]
  change run M (sourcePrefix p r scale ++ suffix p r scale) s=_
  rw [run_append,run_append,prefix_execution]

theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α)
    (p : Math.Layout) (r : Program.Regions) (scale : ℝ) (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (original p r scale) s ↔ TraceSafe M bounds (candidate p r scale) s := by
  unfold TraceSafe
  rw [candidate_body]
  conv_lhs=>rw [←List.take_append_drop 12 (original p r scale).surfaceBody]
  change Safety.traceSafeList M bounds (sourcePrefix p r scale ++ suffix p r scale) s ↔ _
  simp only [Safety.traceSafeList_append,prefix_execution,prefix_safe]

#axiomsClean prefix_execution
#axiomsClean prefix_safe
#axiomsClean execution
#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IndexCSE
