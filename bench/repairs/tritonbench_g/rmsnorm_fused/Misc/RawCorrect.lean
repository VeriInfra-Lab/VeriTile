import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelInterface
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.RmsnormFused.RawCorrect
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole (realAlgebra)
noncomputable section

def kernel_formula (initial : State ℝ) (N : Nat) (epsilon : ℝ) (row col : Nat) : ℝ :=
  Runtime.read initial "X" (row*N+col) *
    (Real.sqrt ((∑ j ∈ Finset.range N,Runtime.read initial "X" (row*N+j)^2)/(N:ℝ)+epsilon))⁻¹ *
      Runtime.read initial "W" col

theorem kernel_real (shape : Shape) (valid : shape.Valid) (epsilon : ℝ)
    (schedule : List Nat) (all : schedule.Perm shape.grid) (initial : State ℝ) :
    ∃ final, Runtime.run realAlgebra "X" "Y" "W" shape.rows shape.width shape.block
      epsilon schedule initial=some final ∧
      (∀ row col,row<shape.rows → col<shape.width →
        Runtime.read final "Y" (row*shape.width+col)=kernel_formula initial shape.width epsilon row col) ∧
      (∀ (region : RegionName) offset,region≠"Y" ∨ shape.rows*shape.width≤offset →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,ran,values,frame⟩ := KernelInterface.complete realAlgebra shape valid epsilon schedule all initial
  refine ⟨final,ran,?_,frame⟩
  intro row col hr hc
  rw [values row col hr hc]
  exact Ideal.value_real initial shape.width shape.block row col epsilon hc valid.2.1

#specNonCircular kernel_formula avoiding [source,Runtime.value,Runtime.run,FP.Structural.exec]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.RawCorrect
