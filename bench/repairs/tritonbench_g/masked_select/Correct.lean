import bench.repairs.tritonbench_g.masked_select.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.MaskedSelect.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole
open scoped LogicalDomainIO
noncomputable section

/-- The complete selected grid produces the stable selected sequence on the
original prepared-prefix domain. Capacity, active extent and exact frame remain. -/
specification masked_select_stable_correctness (rest : Algebra ℝ) (p : Parameters)
    (config : Config) (chosen : config ∈ configs) (separate : KernelSeparate p) :
    Spec.Real (ReaderIO.view rest p config ⊨{ReaderIO.prefixes p} fun xs j => (((List.range p.size).filter xs.selected).map xs.value)[j.val]?.getD (0 : ℝ)) :=
  ReaderProof.correct rest p config chosen separate

#axiomsClean masked_select_stable_correctness
end
end VeriTile.Bench.Repairs.MaskedSelect.Correct
