import bench.tritonbench_g.matrix_reduction.IO
import bench.tritonbench_g.matrix_reduction.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.ProgramPair
open VeriTile.Triton FP.Structural TypedSource ConcreteCorrect FPCorrect
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Two actual complete one-block programs, with source safety, every typed
row output and both exterior frames. No payload or numerical-law restriction. -/
theorem equivalent {α : Type} [Inhabited α] (A : Algebra α)
    (storage : Storage) (M N : Nat) (hN : 0 < N) :
    (matrixReductionNativeIO A storage M N).Equivalent
      (matrixReductionCandidate A storage M N) := by
  intro vals _ bounds bounded initial _
  let original : Invocation α := {
    kernel := kernel storage "x" "out" N 1 1 M N
    backend := A
    pids := fun _ => 0 }
  let s := original.initial initial
  obtain ⟨left, ran, values, frame⟩ := KernelFP.native_graph A storage s M N hN
  obtain ⟨left', ran', mem⟩ := KernelFP.source_graph A storage s M N hN
  obtain ⟨right, cran, cmem⟩ := Candidate.source_graph A storage s M N hN
  have eq : left'=left := Option.some.inj (ran'.symm.trans ran)
  subst left'
  have same : right.mem=left.mem := cmem.trans mem.symm
  have framed : (matrixReductionNativeIO A storage M N).Frame vals initial left := by
    intro r o outside
    have cell : left.mem r o=s.mem r o := by
      apply frame r o
      by_cases hr : r ≠ ("out" : RegionName)
      · exact Or.inl hr
      · right
        by_contra ho
        have bound : o < M := by omega
        have h := outside.1 (0 : Fin 1) (⟨o,bound⟩,()) trivial
        exact h.elim hr (fun ne => ne rfl)
    exact cell.trans (congrFun (congrFun (Invocation.initial_mem original initial) r) o)
  refine ⟨?_,?_,left,right,?_,?_,?_,framed,?_⟩
  · change ScheduleSafe bounds [_] initial
    rw [scheduleSafe_singleton]
    apply KernelSafety.trace A storage M N hN
    · intro i j
      exact bounded.2.1 (0 : Fin 1) (i,j,()) trivial
    · intro i
      exact bounded.2.2.1 (0 : Fin 1) (i,()) trivial
  · change ScheduleSafe bounds [_] initial
    rw [scheduleSafe_singleton]
    apply Candidate.trace A storage M N hN
    · intro i j
      exact bounded.2.1 (0 : Fin 1) (i,j,()) trivial
    · intro i
      exact bounded.2.2.1 (0 : Fin 1) (i,()) trivial
  · simpa only [matrixReductionNativeIO,runSchedule_singleton,Invocation.run] using ran
  · simpa only [matrixReductionCandidate,runSchedule_singleton,Invocation.run] using cran
  · intro o i _
    change mapCell id (left.mem _ _) = mapCell id (right.mem _ _)
    rw [same]
  · intro r o outside
    rw [same]
    exact framed r o outside

#axiomsClean equivalent
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.ProgramPair
