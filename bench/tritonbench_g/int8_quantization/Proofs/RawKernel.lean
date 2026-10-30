import bench.tritonbench_g.int8_quantization.Proofs.SourceObservation

namespace VeriTile.Bench.Repairs.Int8Quantization.RawKernel
open VeriTile.Triton FP.Structural Revision002
noncomputable section
variable {α : Type} [Inhabited α]

/-- Addresses outside every enabled output and scale write. -/
def Frame (p : Parameters) (entry final : State α) : Prop :=
  ∀ r o,
    (r≠scaleRegion ∨ ∀ w∈Schedule.programs 0 (p.batches*p.blocks),o≠w.index) →
    (r≠outRegion ∨ ∀ w∈Schedule.programs 0 (p.batches*p.blocks),∀ i,Grid.active p w i → o≠Grid.offset p w i) →
    final.mem r o=entry.mem r o

/-- The existing full-grid execution/readback/frame proof over raw state.
Zero extents are allowed; positivity of channels and block count is derived
only when an active output coordinate is present. -/
theorem complete (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0<p.width) (block : 0<p.blockRows) (entry : State α) :
    ∃ final,Grid.launch M ops p entry=some final ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) → ∀ i : Lane p,Grid.active p w i →
        final.mem outRegion (Grid.offset p w i)=
          .mk .int (quantized M ops (Grid.parameters p w) positive (Grid.atProgram entry w) i)) ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) →
        final.mem scaleRegion w.index=
          .mk .real (savedScale M ops (Grid.parameters p w) positive (Grid.atProgram entry w))) ∧
      Frame p entry final := by
  refine ⟨Grid.model M ops p positive (Schedule.programs 0 (p.batches*p.blocks)) entry,
    Grid.whole_source M ops p positive entry,?_,?_,Grid.frame M ops p positive _ entry⟩
  · intro w member i active
    have channels : 0<p.channels := by
      by_contra h
      have zero : p.channels=0 := by omega
      have bound := active.1
      simp [zero] at bound
    have blocks : 0<p.blocks := by
      have bound := (Schedule.program_bounds 0 (p.batches*p.blocks) w member).2.2
      by_contra h
      have zero : p.blocks=0 := by omega
      simp [zero] at bound
    exact Grid.readback M ops p positive block channels blocks _ (Schedule.unique _ _) entry w member i active
  · exact fun w member => Grid.scale_readback M ops p positive _ (Schedule.unique _ _) entry w member

#axiomsClean complete
#specNonCircular Frame avoiding [source,FP.Structural.exec,Grid.model,Grid.launch]
end
end VeriTile.Bench.Repairs.Int8Quantization.RawKernel
