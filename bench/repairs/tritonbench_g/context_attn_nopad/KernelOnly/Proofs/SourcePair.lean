import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Kernels
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPReferenceProgram
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.SourcePair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (reused : Bool) (p : Layout) (r : Regions) (scale : ℝ) :=
  if reused then blockStartReusedFP32 p r scale else originalFP32 p r scale

/-- Complete direct original preserves the historical FP32 program. -/
theorem original_preserved (p : Layout) (r : Regions) (scale : ℝ) :
    originalFP32 p r scale =
      VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program.kernel p r .fp32 Bool.false scale := rfl
private def before (p : Layout) (r : Regions) (scale : ℝ) :=
  (originalFP32 p r scale).surfaceBody.take 8
private def indexCode (reused : Bool) (p : Layout) (r : Regions) (scale : ℝ) :=
  ((source reused p r scale).surfaceBody.drop 8).take 1
private def after (p : Layout) (r : Regions) (scale : ℝ) :=
  (originalFP32 p r scale).surfaceBody.drop 9

private theorem split_source (reused : Bool) (p : Layout) (r : Regions) (scale : ℝ) :
    (source reused p r scale).surfaceBody =
      before p r scale ++ (indexCode reused p r scale ++ after p r scale) := by
  cases reused <;> rfl

private theorem prefix_registers {α : Type} [Inhabited α] (M : Algebra α)
    (p : Layout) (r : Regions) (scale : ℝ) (s t : State α)
    (executed : run M (before p r scale) s=some t) :
    t.regs .nat [] "start_m"=some (fun _ => s.pids 2) ∧
      t.regs .nat [] "block_start_loc"=some (fun _ => p.BM*s.pids 2) := by
  simp (config := {maxSteps := 1000000}) [before,originalFP32,
    ComputeKernel.surfaceBody,List.take,run,step,evalExpr,evalOp_unfold,Option.bind,
    numeric,integralFloorDiv,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    ofFloat,toFloat,Region.cast,State.setReg] at executed
  subst t
  simp [State.setReg,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

private theorem index_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : Layout) (r : Regions) (scale : ℝ) (s t : State α)
    (executed : run M (before p r scale) s=some t) :
    run M (indexCode Bool.false p r scale) t =
      run M (indexCode Bool.true p r scale) t := by
  obtain ⟨start,block⟩ := prefix_registers M p r scale s t executed
  simp (config := {maxSteps := 1000000}) [indexCode,source,originalFP32,blockStartReusedFP32,ComputeKernel.surfaceBody,List.drop,List.take,
    run,step,evalExpr,evalOp_unfold,Option.bind,numeric,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,start,block,Nat.mul_comm]
  congr 1
  funext i
  simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]

private theorem index_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (reused : Bool)
    (p : Layout) (r : Regions) (scale : ℝ) (s : State α) :
    Safety.traceSafeList M bounds (indexCode reused p r scale) s := by
  cases reused
  all_goals simp [indexCode,source,originalFP32,blockStartReusedFP32,
    ComputeKernel.surfaceBody,List.drop,List.take,Safety.traceSafeList,
    Safety.stepSafe,Safety.safeExpr,Safety.safeAlg,Safety.safeOp,Safety.safeOp_unfold]
  all_goals split <;> trivial

/-- Reuse the already computed scalar block origin. All floating instructions,
success/failure and the complete final state are preserved for arbitrary words. -/
theorem same_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : Layout) (r : Regions) (scale : ℝ) (s : State α) :
    exec M (originalFP32 p r scale) s = exec M (blockStartReusedFP32 p r scale) s := by
  change run M (source Bool.false p r scale).surfaceBody s =
    run M (source Bool.true p r scale).surfaceBody s
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  simp only [run_append]
  cases h : run M (before p r scale) s with
  | none => rfl
  | some t => simp only [Option.bind_some,index_run M p r scale s t h]

/-- The replacement is a pure index assignment; equal reached states retain
every later mask/address check, without adding an execution premise. -/
theorem same_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (p : Layout) (r : Regions) (scale : ℝ) (s : State α) :
    TraceSafe M bounds (originalFP32 p r scale) s ↔ TraceSafe M bounds (blockStartReusedFP32 p r scale) s := by
  have tail : ∀ t, run M (before p r scale) s=some t →
      (Safety.traceSafeList M bounds (indexCode Bool.false p r scale ++ after p r scale) t ↔
      Safety.traceSafeList M bounds (indexCode Bool.true p r scale ++ after p r scale) t) := by
    intro t executed
    simp only [Safety.traceSafeList_append,index_safe,true_and,
      index_run M p r scale s t executed]
  change Safety.traceSafeList M bounds (source Bool.false p r scale).surfaceBody s ↔
    Safety.traceSafeList M bounds (source Bool.true p r scale).surfaceBody s
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  constructor
  · rintro ⟨head,next⟩
    exact ⟨head,fun t h => (tail t h).mp (next t h)⟩
  · rintro ⟨head,next⟩
    exact ⟨head,fun t h => (tail t h).mpr (next t h)⟩

#axiomsClean original_preserved
#axiomsClean same_run
#axiomsClean same_safe
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.SourcePair
