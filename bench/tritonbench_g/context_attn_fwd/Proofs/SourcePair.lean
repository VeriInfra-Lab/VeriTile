import bench.tritonbench_g.context_attn_fwd.Kernels
import bench.tritonbench_g.context_attn_fwd.Proofs.FPSuccess
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ContextAttnFwd.SourcePair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (candidate : Bool) (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  if candidate then blockEndFactoredFP32 p r scale else originalFP32 p r scale

/-- Complete original preservation, independently of the new integer rewrite. -/
theorem original_preserved (p : Layout) (r : Regions) (scale : ℝ) :
    originalFP32 p r scale =
      VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp32 Bool.false scale := rfl

private def position : Nat := 18
private def indexCode (candidate : Bool) (p : Layout) (r : Regions) (scale : ℝ) :=
  ((source candidate p r scale).surfaceBody.drop position).take 1

private theorem index_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : Layout) (r : Regions) (scale : ℝ) (s : State α) :
    run M (indexCode Bool.false p r scale) s = run M (indexCode Bool.true p r scale) s := by
  cases hb : s.regs .nat [] "block_start_loc" <;>
    cases hn : s.regs .int [] "cur_batch_seq_len" <;>
    cases hp : s.regs .int [] "prompt_cache_len"
  all_goals simp (config := {maxSteps := 1000000}) [indexCode,position,source,
    originalFP32,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original,VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original,blockEndFactoredFP32,ComputeKernel.surfaceBody,List.drop,List.take,
    run,step,evalExpr,evalOp_unfold,Option.bind,numeric,numericLt,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,hb,hn,hp]
  all_goals try (congr 1; funext i; simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.add_lt_add_iff_right]; split <;> rfl)

private theorem index_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (candidate : Bool) (p : Layout) (r : Regions) (scale : ℝ) (s : State α) :
    Safety.traceSafeList M bounds (indexCode candidate p r scale) s := by
  cases candidate
  all_goals simp [indexCode,position,source,originalFP32,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original,VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original,blockEndFactoredFP32,
    ComputeKernel.surfaceBody,List.drop,List.take,Safety.traceSafeList,Safety.stepSafe,
    Safety.safeExpr,Safety.safeAlg,Safety.safeOp,Safety.safeOp_unfold]
  all_goals split <;> trivial

private theorem split_source (candidate : Bool) (p : Layout) (r : Regions) (scale : ℝ) :
    (source candidate p r scale).surfaceBody =
      (source candidate p r scale).surfaceBody.take position ++
        (indexCode candidate p r scale ++ (source candidate p r scale).surfaceBody.drop (position+1)) := by
  cases candidate <;> rfl
private theorem prefix_eq (p : Layout) (r : Regions) (scale : ℝ) :
    (source Bool.false p r scale).surfaceBody.take position =
      (source Bool.true p r scale).surfaceBody.take position := rfl
private theorem suffix_eq (p : Layout) (r : Regions) (scale : ℝ) :
    (source Bool.false p r scale).surfaceBody.drop (position+1) =
      (source Bool.true p r scale).surfaceBody.drop (position+1) := rfl

/-- Strict integer common-addend factoring preserves the entire source state;
there is no floating numerical rule or input-word restriction. -/
theorem same_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : Layout) (r : Regions) (scale : ℝ) (s : State α) :
    exec M (originalFP32 p r scale) s = exec M (blockEndFactoredFP32 p r scale) s := by
  change run M (source Bool.false p r scale).surfaceBody s = run M (source Bool.true p r scale).surfaceBody s
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  simp only [run_append,prefix_eq,index_run,suffix_eq]

theorem same_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (p : Layout) (r : Regions) (scale : ℝ) (s : State α) :
    TraceSafe M bounds (originalFP32 p r scale) s ↔ TraceSafe M bounds (blockEndFactoredFP32 p r scale) s := by
  have tail : ∀ t,
      Safety.traceSafeList M bounds (indexCode Bool.false p r scale ++
        (source Bool.false p r scale).surfaceBody.drop (position+1)) t ↔
      Safety.traceSafeList M bounds (indexCode Bool.true p r scale ++
        (source Bool.true p r scale).surfaceBody.drop (position+1)) t := by
    intro t
    simp only [Safety.traceSafeList_append,suffix_eq,index_run,index_safe,true_and]
  change Safety.traceSafeList M bounds (source Bool.false p r scale).surfaceBody s ↔
    Safety.traceSafeList M bounds (source Bool.true p r scale).surfaceBody s
  conv_lhs => rw [split_source]
  conv_rhs => rw [split_source]
  simp only [Safety.traceSafeList_append,prefix_eq,tail]
#axiomsClean original_preserved
#axiomsClean same_run
#axiomsClean same_safe
end
end VeriTile.Bench.Repairs.ContextAttnFwd.SourcePair
