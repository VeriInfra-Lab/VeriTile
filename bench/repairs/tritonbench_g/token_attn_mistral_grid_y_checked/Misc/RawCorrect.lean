import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Kernels
import bench.repairs.tritonbench_g.token_attn_mistral.Real
import bench.repairs.tritonbench_g.token_attn_mistral.Stores

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.RawCorrect
open VeriTile.Triton FP.Structural TokenAttnMistral
noncomputable section

/-- Complete shared-source grid execution over raw typed kernel memory.
The independent page-table weighted sum and exact exterior frame hold for all
submitted events. Strides, regions, geometry and ownership are kernel facts. -/
theorem kernel_correct (p : Parameters) (work : List Grid.PID) (memory : Grid.Memory ℝ)
    (positive : 0 < p.block) (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    ∃ final, Grid.run Real.algebra Real.sum p work memory = some final ∧
      (∀ e ∈ Grid.events p work,
        (final p.output (Grid.offset p e.1 e.2)).read .fp32 =
          Real.weighted p (Grid.state memory e.1) e.2.1.val) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ e ∈ Grid.events p work, o ≠ Grid.offset p e.1 e.2) →
        final r o = memory r o) := by
  obtain ⟨final,run,_,values,frame⟩ := Grid.whole Real.algebra Real.sum p work memory separate unique
  refine ⟨final,run,?_,frame⟩
  intro e he
  rw [values e he,Cell.read_same]
  exact Real.output_weighted p (Grid.state memory e.1) positive e.2

#axiomsClean kernel_correct
#specNonCircular Real.weighted avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.RawCorrect
