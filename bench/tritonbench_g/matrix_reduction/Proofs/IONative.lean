import bench.tritonbench_g.matrix_reduction.IO
import bench.tritonbench_g.matrix_reduction.Proofs.KernelSafety

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.IONative
open VeriTile.Triton FP.Structural TypedSource ConcreteCorrect FPCorrect
noncomputable section

/-- Successful source-to-native observation with actual access safety and
 the complete typed output/exterior frame. No second kernel or floating rewrite
is asserted: the scalar primitive interpretation is the model boundary. -/
theorem correct {α : Type} [Inhabited α] (A : Algebra α)
    (storage : Storage) (M N : Nat) (hN : 0 < N) :
    (matrixReductionNativeIO A storage M N).Implements (nativeRowMaximum A storage M N) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial _
    change ScheduleSafe bounds [_] initial
    rw [scheduleSafe_singleton]
    apply KernelSafety.trace A storage M N hN
    · intro i j
      exact bounded.2.1 (0 : Fin 1) (i, j, ()) trivial
    · intro i
      exact bounded.2.2.1 (0 : Fin 1) (i, ()) trivial
  · intro vals initial input
    let job : Invocation α := {
      kernel := kernel storage "x" "out" N 1 1 M N
      backend := A
      pids := fun _ => 0 }
    obtain ⟨final, run, values, frame⟩ := KernelFP.native_graph A storage
      (job.initial initial) M N hN
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [matrixReductionNativeIO, runSchedule_singleton, Invocation.run] using run
    · intro o i _
      have inputs : KernelFP.sourceValues storage M N (job.initial initial) =
          fun row col => toFloat (channel storage) (vals (0 : Fin 1) (row, col, ())) := by
        funext row col
        have binding := input (0 : Fin 1) (row, col, ()) trivial
        cases storage <;>
          simpa [KernelFP.sourceValues, job, Invocation.initial, bindRegisters,
            matrixReductionNativeIO, channel, mapValue, toFloat] using binding
      fin_cases o
      change mapCell id (final.mem "out" i.1.val) = _
      rw [values, inputs]
      cases storage <;> rfl
    · intro r o outside
      apply frame r o
      by_cases hr : r ≠ ("out" : RegionName)
      · exact Or.inl hr
      · right
        by_contra ho
        have bound : o < M := by omega
        have h := outside.1 (0 : Fin 1) (⟨o, bound⟩, ()) trivial
        exact h.elim hr (fun ne => ne rfl)

#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.IONative
