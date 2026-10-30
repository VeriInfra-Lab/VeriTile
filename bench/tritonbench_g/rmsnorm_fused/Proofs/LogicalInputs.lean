import bench.tritonbench_g.rmsnorm_fused.Proofs.KernelInterface
import bench.tritonbench_g.rmsnorm_fused.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.RmsnormFused.KernelLogical
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole (realAlgebra)
noncomputable section

/-- Independent RMSNorm on logical input arrays, with no source-memory reader. -/
def formula (N : Nat) (epsilon : ℝ) (xs : Nat → Nat → ℝ) (weights : Nat → ℝ)
    (row col : Nat) : ℝ :=
  xs row col * (Real.sqrt ((∑ j ∈ Finset.range N, (xs row j)^2)/(N:ℝ)+epsilon))⁻¹ *
    weights col

/-- Connect the existing complete source result to bounded logical arrays.
This operational helper is the input/output adapter for the public grid IO. -/
theorem complete (shape : Shape) (valid : shape.Valid) (epsilon : ℝ)
    (schedule : List Nat) (all : schedule.Perm shape.grid) (initial : State ℝ)
    (xs : Nat → Nat → ℝ) (weights : Nat → ℝ)
    (input : ∀ row col, row<shape.rows → col<shape.width →
      Runtime.read initial "X" (row*shape.width+col)=xs row col)
    (weight : ∀ col, col<shape.width → Runtime.read initial "W" col=weights col) :
    ∃ final, Runtime.run realAlgebra "X" "Y" "W" shape.rows shape.width shape.block
      epsilon schedule initial=some final ∧
      (∀ row col,row<shape.rows → col<shape.width →
        Runtime.read final "Y" (row*shape.width+col)=formula shape.width epsilon xs weights row col) ∧
      (∀ (region : RegionName) offset,region≠"Y" ∨ shape.rows*shape.width≤offset →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,ran,values,frame⟩ := KernelInterface.complete realAlgebra shape valid epsilon schedule all initial
  refine ⟨final,ran,?_,frame⟩
  intro row col hr hc
  rw [values row col hr hc,Ideal.value_real initial shape.width shape.block row col epsilon hc valid.2.1]
  have rowSum : (∑ j ∈ Finset.range shape.width, Runtime.read initial "X" (row*shape.width+j)^2) =
      ∑ j ∈ Finset.range shape.width, (xs row j)^2 := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [input row j hr (Finset.mem_range.mp hj)]
  rw [rowSum,input row col hr hc,weight col hc]
  rfl

#specNonCircular formula avoiding [source,Runtime.value,Runtime.run,FP.Structural.exec]
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.KernelLogical
