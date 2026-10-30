import bench.repairs.tritonbench_g.mean_reduction.IO
import bench.repairs.tritonbench_g.mean_reduction.Proofs.KernelRealResult

namespace VeriTile.Bench.Repairs.MeanReduction.MeanIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MeanReduction
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

theorem realLaunch_initial (pid : Nat) (s : BlockState) :
    (realLaunch pid).initial s=Whole.tileState pid s := by
  simp [realLaunch,AlgorithmLaunch.initial,Whole.tileState]

theorem real_schedule_run (M N : Nat) (jobs : List Nat) (s : BlockState) :
    runAlgorithmSchedule (realIO M N).kernel.toAlgKernel (jobs.map realLaunch) s=
      Whole.runTiles "X" "Mean" M N jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons,runAlgorithmSchedule,AlgorithmLaunch.run,realLaunch_initial,
      Whole.runTiles]
    change (exec (Whole.source "X" "Mean" M N).eraseDType.toAlgKernel
      (Whole.tileState pid s)).bind _=_
    cases exec (Whole.source "X" "Mean" M N).eraseDType.toAlgKernel (Whole.tileState pid s) with
    | none => rfl
    | some t => exact ih t

/-- Complete arbitrary-order grid result, obtained from the established row
mean proof. Access safety follows every actual source invocation separately. -/
theorem real_implements (M N : Nat) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridSize M))) :
    (realIO M N).RegionalScheduledImplements (realSchedule M N jobs) (mathematical M N) := by
  apply UKernelIO.RegionalScheduledImplements.intro
  · intro s values _ bounds backed
    have input : M*N≤bounds "X" := by
      by_contra hn
      have bad := backed.1 ⟨0,by simp [realIO]⟩ ⟨bounds "X",Nat.lt_of_not_ge hn⟩ trivial
      exact Nat.lt_irrefl (bounds "X") bad
    have output : M≤bounds "Mean" := by
      by_contra hn
      have bad := backed.2.1 ⟨0,by simp [realIO]⟩ ⟨bounds "Mean",Nat.lt_of_not_ge hn⟩ trivial
      exact Nat.lt_irrefl (bounds "Mean") bad
    apply algorithmScheduleSafe_of_forall
    intro launch _ t
    exact Whole.source_trace_safe "X" "Mean" M N (launch.initial t) bounds input output
  · intro s values bound
    obtain ⟨t,ran,result,frame⟩ := Whole.grid_result "X" "Mean" M N (by decide) jobs order s
    refine ⟨t,(real_schedule_run M N jobs s).trans ran,?_,?_⟩
    · intro o j _
      change t.readMem "Mean" j.val=mathematical M N values o j
      rw [result j.val j.isLt]
      unfold Whole.rowMean mathematical
      congr 1
      apply Finset.sum_congr rfl
      intro k _
      exact bound ⟨0,by simp [realIO]⟩ ⟨j.val*N+k.val,rowMajor_bound j.isLt k.isLt⟩ trivial
    · intro r offset outside _
      apply frame r offset
      by_cases hr : r="Mean"
      · right
        by_contra hn
        have missed := outside ⟨0,by simp [realIO]⟩ ⟨offset,Nat.lt_of_not_ge hn⟩ trivial
        exact missed.elim (fun ne => ne hr) (fun ne => ne rfl)
      · exact Or.inl hr

/-- Logical coordinates read the same canonical row-major input cell. -/
theorem matrixInputs_apply (M N : Nat) (values : Fin 1 → Fin (M*N) → ℝ)
    (row : Fin M) (col : Fin N) :
    matrixInputs M N values row col =
      values 0 ⟨row.val*N+col.val, rowMajor_bound row.isLt col.isLt⟩ := by
  change values 0 (finProdFinEquiv (row, col)) = _
  congr 1
  apply Fin.ext
  simp [finProdFinEquiv, Nat.add_comm, Nat.mul_comm]

/-- The logical matrix statement is equivalent to the complete previous grid
judgment, including its input binding, actual schedule, safety and raw frame. -/
theorem logical_mean_iff (M N : Nat) (jobs : List Nat) :
    (meanIO M N jobs ⊨ fun xs row =>
      (∑ col : Fin N, xs row col) / (N : ℝ)) ↔
    (realIO M N).RegionalScheduledImplements (realSchedule M N jobs) (mathematical M N) := by
  change (realIO M N).RegionalScheduledImplements (realSchedule M N jobs)
    (fun values (_ : Fin 1) (row : Fin M) =>
      (∑ col : Fin N, matrixInputs M N values row col) / (N : ℝ)) ↔ _
  have target_eq : (fun values (_ : Fin 1) (row : Fin M) =>
      (∑ col : Fin N, matrixInputs M N values row col) / (N : ℝ)) = mathematical M N := by
    funext values channel row
    simp only [mathematical, matrixInputs_apply]
  rw [target_eq]

theorem logical_implements (M N : Nat) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridSize M))) :
    meanIO M N jobs ⊨ fun xs row => (∑ col : Fin N, xs row col) / (N : ℝ) :=
  (logical_mean_iff M N jobs).mpr (real_implements M N jobs order)

#axiomsClean logical_mean_iff
#axiomsClean logical_implements

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.MeanReduction.MeanIO
