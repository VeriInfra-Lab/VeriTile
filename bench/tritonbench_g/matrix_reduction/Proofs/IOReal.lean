import bench.tritonbench_g.matrix_reduction.IO

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.IOReal
open VeriTile.Triton TypedSource
open scoped VeriTile.Triton.MaskedTileKernelIO₁
noncomputable section

/-- Every output is the greatest element of its nonempty input row. The
shared IO judgment supplies actual source access safety and the complete
flat-memory output/exterior frame. Empty row counts remain admitted. -/
theorem correct (storage : Storage) (M N : Nat) (hN : 0 < N) :
    VeriTile.Spec.Real (matrixReductionIO storage M N ⊨ fun _ xs i =>
      KernelMath.maximumValue hN (fun j => xs (i.1, j, ()))) := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · change ((kernel storage "x" "out" N 1 1 M N).eraseDType.toAlgKernel).FlattenOk
    unfold Kernel.FlattenOk
    rw [KernelReal.projection]
    exact matrix_reduction_flattenOk "x" "out" N 1 1 M N
  · intro bounds s hx hy
    change Kernel.TraceSafe bounds (kernel storage "x" "out" N 1 1 M N).eraseDType.toAlgKernel s
    unfold Kernel.TraceSafe
    rw [KernelReal.projection]
    exact matrix_reduction_traceSafe "x" "out" N 1 1 M N hN bounds s
      (fun i => by simpa [xOffset] using hx i trivial)
      (fun i => hy (i, ⟨0, hN⟩, ()) rfl)
  · intro s xs inputs
    obtain ⟨t, run, values, frame⟩ := KernelReal.coordinates storage M N hN s
      (fun i j => xs (i, j, ())) (fun i j => inputs (i, j, ()) trivial)
    refine ⟨t, run, ?_, ?_⟩
    · intro i _
      exact KernelMath.maximum_eq _ _ _ (values i.1)
        (KernelMath.maximumValue_isMaximum hN _)
    · intro r o outside
      apply frame r o
      rcases outside with hr | ho
      · exact Or.inl hr
      · exact Or.inr (by
          by_contra h
          exact ho (⟨o, by omega⟩, ⟨0, hN⟩, ()) rfl rfl)

#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.IOReal
