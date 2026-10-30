import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Contract

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
set_option maxHeartbeats 3000000

/-- Complete legal launch order, without a fixed sequential PID convention. -/
structure Schedule (r : Whole.Request) where
  order : List Nat
  complete : order.Perm (List.range (Whole.programs r))

abbrev dsl := Whole.contractedKernel
abbrev sourceDsl := Whole.kernel
abbrev run := @Whole.Contracted.runPrograms

theorem scheduled_grid {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Whole.Contracted.Numerics α)
    (law : Whole.Contracted.PrimitiveLaws M D Q) (r : Whole.Request)
    (schedule : Schedule r) (start : State α) (hCA : r.C ≠ r.A) (hCB : r.C ≠ r.B) :
    ∃ final, run M D r schedule.order start = some final ∧
      Whole.Contracted.Result Q r start final := by
  obtain ⟨final, he, _, _, frame, _, value⟩ :=
    Whole.Contracted.programs_correct M D Q law r start start schedule.order
      hCA hCB (fun _ => rfl) (fun _ => rfl)
  refine ⟨final, he, ?_, frame⟩
  intro row hr col hc
  obtain ⟨pid, hp, i, hi, hj⟩ := Whole.cell_covered r start row col hr hc
  exact value row hr col hc
    ⟨pid, schedule.complete.mem_iff.mpr (List.mem_range.mpr hp), i, hi, hj⟩

theorem empty_grid {α : Type} [Inhabited α] (M : Algebra α) (D : FPFused.DotAcc α)
    (r : Whole.Request) (schedule : Schedule r) (s : State α)
    (empty : r.rows = 0 ∨ r.columns = 0) : run M D r schedule.order s = some s := by
  have hp : Whole.programs r = 0 := by
    rcases empty with h | h
    all_goals simp [Whole.programs, Whole.tileRows, Whole.tileColumns, h]
  have hz : schedule.order = [] := by
    have h := schedule.complete
    simpa [hp] using h
  simp [run, hz, Whole.Contracted.runPrograms]

/-- Complete selected DSL execution plus all existing finite source safety
facts, derived from the actual Tensor bindings and executed guards. -/
theorem execute {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Whole.Contracted.Numerics α)
    (law : Whole.Contracted.PrimitiveLaws M D Q)
    (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (h : InputContract (metadata a b)) (activation : String)
    (c : OutputTensor) (cl : OutputLaws c a b) (fresh : Fresh a b c)
    (output : outputCheck (metadata a b) (outputBase c) = Bool.true)
    (schedule : Schedule (request a b activation)) (read : Nat → α) (scratch : State α) :
    ∃ final, run M D (request a b activation) schedule.order (initial a b read scratch) = some final ∧
      Whole.Contracted.Result Q (request a b activation) (initial a b read scratch) final ∧
      Whole.SourceSafety (request a b activation) (initial a b read scratch)
        (tensorContract a b al bl h activation c cl fresh output read scratch) := by
  obtain ⟨final, he, result⟩ := scheduled_grid M D Q law (request a b activation) schedule
    (initial a b read scratch)
    (by change ("c" : RegionName) ≠ "a"; decide)
    (by change ("c" : RegionName) ≠ "b"; decide)
  exact ⟨final, he, result, Whole.source_safety _ (admitted a b al bl h activation).1 _ _⟩

#specNonCircular Whole.Contracted.logicalReference avoiding [run, dsl, sourceDsl]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
