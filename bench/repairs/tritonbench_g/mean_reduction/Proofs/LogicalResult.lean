import bench.repairs.tritonbench_g.mean_reduction.IO
import bench.tritonbench_g.mean_reduction.Proofs.WholeFPFamily

namespace VeriTile.Bench.Repairs.MeanReduction.MeanIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MeanReduction
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

def Reads {α : Type} [Inhabited α] (values : Inputs α M N)
    (memory : RegionName → Nat → Cell α) : Prop :=
  ∀ i : Fin (M*N), (memory "X" i.val).read .real=element values i

theorem accumulated_eq {α : Type} [Inhabited α] (A : Algebra α)
    (M N pid : Nat) (values : Inputs α M N) (memory : RegionName → Nat → Cell α)
    (reads : Reads values memory) (off : Nat) (acc : Accumulator α) :
    WholeFPValue.accumulated A "X" M N pid memory off acc = accumulated A M N pid values off acc := by
  conv_lhs => rw [WholeFPValue.accumulated]
  conv_rhs => rw [accumulated]
  split
  · have stepEq : WholeFPValue.nextAcc A "X" M N pid off memory acc =
        fun i => A.binary (some .fp32) .real .add (acc i) (chunk A M N pid off values i) := by
      funext i
      simp only [WholeFPValue.nextAcc,WholeFPValue.chunk,chunk]
      split
      next active => rw [reads ⟨_,rowMajor_bound active.1 active.2⟩]
      next => rfl
    rw [stepEq]
    exact accumulated_eq A M N pid values memory reads (off+8) _
  · rfl
termination_by N-off

theorem rowValue_eq {α : Type} [Inhabited α] (family : Nat → Algebra α)
    (M N : Nat) (values : Inputs α M N) (memory : RegionName → Nat → Cell α)
    (reads : Reads values memory) (row : Fin M) :
    WholeFPFamily.rowValue family "X" M N memory row.val = nativeRow family M N values row := by
  unfold WholeFPFamily.rowValue WholeFPValue.rowValue WholeFPValue.tileValue
    WholeFPValue.meanValues nativeRow
  rw [accumulated_eq _ M N _ values memory reads]
  rfl

theorem invocation_initial {α : Type} (family : Nat → Algebra α) (variant : Bool)
    (M N pid : Nat) (s : State α) :
    (invocation family variant M N pid).initial s=WholeFP.tileState pid s := by
  simp [Invocation.initial,invocation,bindRegisters,WholeFP.tileState]

theorem schedule_run {α : Type} [Inhabited α] (family : Nat → Algebra α) (variant : Bool)
    (M N : Nat) (jobs : List Nat) (s : State α) :
    runSchedule (jobs.map (invocation family variant M N)) s=
      WholeFPFamily.runTiles family variant "X" "Mean" M N jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,invocation_initial,WholeFPFamily.runTiles]
    have kernelEq : (invocation family variant M N pid).kernel=
        typedKernel .fp32 .fp32 variant "X" "Mean" M N 8 8 := by cases variant <;> rfl
    rw [kernelEq]
    change (exec (family pid) (typedKernel .fp32 .fp32 variant "X" "Mean" M N 8 8)
      (WholeFP.tileState pid s)).bind _=_
    cases exec (family pid) (typedKernel .fp32 .fp32 variant "X" "Mean" M N 8 8)
      (WholeFP.tileState pid s) with
    | none => rfl
    | some t => exact ih t

/-- The existing whole-grid execution is reused, with only its input and
output views changed from cells to finite logical arrays. -/
theorem complete_result {α : Type} [Inhabited α] (family : Nat → Algebra α)
    (variant : Bool) (M N : Nat) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridSize M))) (values : Inputs α M N) (s : State α)
    (reads : Reads values s.mem) :
    ∃ t, runSchedule (jobs.map (invocation family variant M N)) s=some t ∧
      (∀ i : Fin M,t.mem "Mean" i.val=Cell.mk .real (nativeRow family M N values i)) ∧
      (∀ (r : RegionName) o,r≠"Mean" ∨ M≤o → t.mem r o=s.mem r o) := by
  obtain ⟨t,ran,result,frame⟩ := WholeFPFamily.grid_result family "X" "Mean" M N
    (by decide) jobs order s
  refine ⟨t,?_,?_,frame⟩
  · rw [schedule_run]
    cases variant
    · exact ran
    · exact (WholeFPFamily.grid_equal family "X" "Mean" M N jobs s).symm.trans ran
  · intro i
    rw [result i.val i.isLt,rowValue_eq family M N values s.mem reads i]

#axiomsClean complete_result
end
end VeriTile.Bench.Repairs.MeanReduction.MeanIO
