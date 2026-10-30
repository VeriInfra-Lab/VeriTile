import bench.repairs.tritonbench_g.parallel_retention_attention.Kernels
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicSourceMemory
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
open VeriTile Triton FP.Structural Kernel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (ForwardConfig BackwardConfig)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Proof-only contextual equality, including actual predecessor-state safety. -/
structure CodeEq (M : Algebra α) (a b : List ComputeStmt) : Prop where
  run : ∀ s, FP.Structural.run M a s=FP.Structural.run M b s
  safe : ∀ cap s, Safety.traceSafeList M cap a s ↔ Safety.traceSafeList M cap b s

structure StmtEq (M : Algebra α) (a b : ComputeStmt) : Prop where
  step : ∀ s, FP.Structural.step M a s=FP.Structural.step M b s
  safe : ∀ cap s, Safety.stepSafe M cap a s ↔ Safety.stepSafe M cap b s

theorem CodeEq.refl (M : Algebra α) (a : List ComputeStmt) : CodeEq M a a :=
  ⟨fun _=>rfl,fun _ _=>Iff.rfl⟩

theorem CodeEq.append {M : Algebra α} {a b c d : List ComputeStmt}
    (h : CodeEq M a b) (k : CodeEq M c d) : CodeEq M (a++c) (b++d) := by
  constructor
  · intro s
    simp only [run_append,h.run]
    cases FP.Structural.run M b s <;> simp [Option.bind,k.run]
  · intro cap s
    rw [Safety.traceSafeList_append,Safety.traceSafeList_append,h.safe]
    simp only [h.run,k.safe]

theorem StmtEq.singleton {M : Algebra α} {a b : ComputeStmt} (h : StmtEq M a b) :
    CodeEq M [a] [b] := by
  constructor
  · intro s
    simp only [run,h.step]
  · intro cap s
    simp only [Safety.traceSafeList_cons,h.safe]
    simp

theorem range_context (M : Algebra α) (a b : List ComputeStmt) (h : CodeEq M a b)
    (idx : RegName) (cur stop stride : Nat) (s : State α) :
    FP.Structural.range M idx cur stop stride a s=FP.Structural.range M idx cur stop stride b s ∧
    ∀ cap,Safety.rangeSafe M cap idx cur stop stride a s ↔
      Safety.rangeSafe M cap idx cur stop stride b s := by
  by_cases hz : stride=0
  · constructor
    · rw [FP.Structural.range,FP.Structural.range]; simp only [hz,if_true]
    · intro cap; rw [Safety.rangeSafe,Safety.rangeSafe]; simp only [hz,if_true]
  by_cases hl : cur<stop
  · constructor
    · rw [FP.Structural.range,FP.Structural.range]
      simp only [hz,hl,if_false,if_true,h.run]
      cases he : FP.Structural.run M b (s.setReg idx .nat [] (fun _=>cur)) with
      | none=>rfl
      | some t=>exact (range_context M a b h idx (cur+stride) stop stride t).1
    · intro cap
      rw [Safety.rangeSafe,Safety.rangeSafe]
      simp only [hz,hl,if_false,if_true,h.run]
      cases he : FP.Structural.run M b (s.setReg idx .nat [] (fun _=>cur)) with
      | none=>exact and_congr (h.safe cap _) Iff.rfl
      | some t=>exact and_congr (h.safe cap _) ((range_context M a b h idx (cur+stride) stop stride t).2 cap)
  · constructor
    · rw [FP.Structural.range,FP.Structural.range]; simp only [hz,hl,if_false]
    · intro cap; rw [Safety.rangeSafe,Safety.rangeSafe]; simp only [hz,hl,if_false]
termination_by stop-cur
decreasing_by all_goals omega

theorem static_context {M : Algebra α} {a b : List ComputeStmt} (h : CodeEq M a b)
    (idx : RegName) (first last stride : Nat) :
    StmtEq M (.forRange idx first last stride a) (.forRange idx first last stride b) := by
  constructor
  · intro s
    simpa only [FP.Structural.step] using (range_context M a b h idx first last stride s).1
  · intro cap s
    simpa only [Safety.stepSafe_unfold] using (range_context M a b h idx first last stride s).2 cap

theorem dynamic_context {M : Algebra α} {a b : List ComputeStmt} (h : CodeEq M a b)
    (idx : RegName) (first last stride : Op .nat []) :
    StmtEq M (.forRangeDyn idx first last stride a) (.forRangeDyn idx first last stride b) := by
  constructor
  · intro s
    simp only [step]
    cases evalOp M none first s <;> cases evalOp M none last s <;> cases evalOp M none stride s <;>
      simp [Option.bind,(range_context M a b h idx _ _ _ s).1]
  · intro cap s
    simp only [Safety.stepSafe_unfold]
    cases evalOp M none first s <;> cases evalOp M none last s <;> cases evalOp M none stride s <;>
      simp [(range_context M a b h idx _ _ _ s).2 cap]

theorem conditional_context {M : Algebra α} {a b : List ComputeStmt} (h : CodeEq M a b)
    (cond : Op .bool []) : StmtEq M (.ifThen cond a) (.ifThen cond b) := by
  constructor
  · intro s
    simp only [step]
    cases evalOp M none cond s <;> simp [Option.bind,h.run]
  · intro cap s
    simp only [Safety.stepSafe_unfold]
    cases evalOp M none cond s <;> simp [h.safe]

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
