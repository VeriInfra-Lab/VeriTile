import bench.tritonbench_g.matrix_reduction.Proofs.KernelFP
import bench.tritonbench_g.matrix_reduction.Proofs.ConcreteCorrect

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.KernelFP
open VeriTile.Triton FP.Structural TypedSource ConcreteCorrect
noncomputable section
/-- Exact source execution in the explicit binary16/binary32 backend. All
 words are executable; the non-NaN maximum interpretation is a separate lemma. -/
theorem all_words (d : Storage) (rest : Algebra Word) (s : State Word)
    (M N : Nat) (hN : 0 < N) :
    ∃ t, exec (backend rest) (kernel d "x" "out" N 1 1 M N) s = some t ∧
      (∀ i : Fin M, t.mem "out" i.val =
        .mk (channel d).toTileDType (ofFloat (channel d) (output d s "x" N 1 M N i))) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run d rest s "x" "out" N 1 1 M N hN
  exact ⟨t, he, fun i => hm ▸ written_readback d s "x" "out" N 1 M N i,
    fun r o outside => hm ▸ written_frame d s "x" "out" N 1 M N r o outside⟩

#axiomsClean all_words
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.KernelFP
