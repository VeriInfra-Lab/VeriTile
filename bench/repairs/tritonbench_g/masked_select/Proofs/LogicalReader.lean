import bench.repairs.tritonbench_g.masked_select.IO
import bench.repairs.tritonbench_g.masked_select.Proofs.PublicIO
namespace VeriTile.Bench.Repairs.MaskedSelect.ReaderProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole
open ReaderIO
open scoped LogicalDomainIO
noncomputable section

/-- The entire old exact-cell judgment is equivalent, with the same prefix
relation, dynamic output extent, schedule, safety and exterior frame. -/
theorem judgment_iff (rest : Algebra ℝ) (p : Parameters) (config : Config) :
    (view rest p config ⊨{prefixes p} fun xs j => (((List.range p.size).filter xs.selected).map xs.value)[j.val]?.getD (0 : ℝ)) ↔
      (IO.selectionIO rest p config).ImplementsOnCells (IO.Prefixes p) (IO.mathematical p) := by
  rfl

theorem correct (rest : Algebra ℝ) (p : Parameters) (config : Config)
    (chosen : config ∈ configs) (separate : KernelSeparate p) :
    (view rest p config ⊨{prefixes p} fun xs j => (((List.range p.size).filter xs.selected).map xs.value)[j.val]?.getD (0 : ℝ)) :=
  (judgment_iff rest p config).mpr (PublicIO.implements rest p config chosen separate)
end
end VeriTile.Bench.Repairs.MaskedSelect.ReaderProof
