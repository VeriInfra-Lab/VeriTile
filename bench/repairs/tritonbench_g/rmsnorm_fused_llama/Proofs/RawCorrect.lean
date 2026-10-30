import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelCandidate
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelReference

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RawCorrect
open VeriTile Triton
noncomputable section

/-- Full Real source and index-commuted candidate grids on arbitrary raw
memory, with the independent RMS formula and every exterior cell preserved. -/
theorem rmsnorm_kernel_real (X Y W : RegionName) (rows columns : Nat) (epsilon : ℝ)
    (initial : BlockState) (xy : X≠Y) (wy : W≠Y) :
    ∃ final,
      RealGrid.run X Y W rows columns 16384 epsilon initial rows=some final ∧
      KernelCandidate.realGrid X Y W rows columns 16384 epsilon initial rows=some final ∧
      (∀ row col,row<rows → col<columns →
        final.readMem Y (row*columns+col)=
          RealGrid.value (initial.readMem X) (initial.readMem W) row columns epsilon col) ∧
      RealGrid.Frame Y (rows*columns) initial final := by
  obtain ⟨final,ran,values,frame⟩ := RealGrid.complete X Y W rows columns 16384 epsilon initial
    (by decide) xy wy rows
  exact ⟨final,ran,by rw [KernelCandidate.real_grid_eq]; exact ran,
    fun row col hr hc=>values row hr col hc,frame⟩

alias denominator_positive := KernelReference.denominator_positive
#axiomsClean rmsnorm_kernel_real
#specNonCircular RealGrid.value avoiding [kernel,RealGrid.run]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RawCorrect
