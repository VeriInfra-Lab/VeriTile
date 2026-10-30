import bench.tritonbench_g.int8_quantization.Proofs.RawKernel
import bench.tritonbench_g.int8_quantization.Proofs.SourceRealTotal

namespace VeriTile.Bench.Repairs.Int8Quantization.Specification
open VeriTile.Triton FP.Structural Revision002
noncomputable section

/-- Independent total ideal-Real target: invalid blocks produce zero; valid
blocks use max-absolute scaling and nearest-away rounding. -/
def totalOutput (p : Parameters) (entry : State ℝ) (i : Lane p) : Int := by
  classical
  exact if Math.FiniteDomain p entry then
    RealSpec.nearestAway (127*(RealSpec.factor p.query p.channels*Math.raw p entry i)/Math.blockMaximum p entry)
  else 0

def totalScale (p : Parameters) (entry : State ℝ) : ℝ := by
  classical
  exact if Math.FiniteDomain p entry then (RealSpec.target p.query p.channels (Math.logicalBlock p entry)).2 else 0

/-- Complete mathematical kernel execution, all enabled outputs and scales,
and memory frame, without Tensor objects or a finite-input restriction. -/
theorem per_block_int8_real_raw (p : Parameters) (positive : 0<p.width)
    (block : 0<p.blockRows) (factor : p.factor=RealSpec.factor p.query p.channels)
    (rest : Algebra ℝ) (entry : State ℝ) :
    ∃ final,Grid.launch (Math.algebra rest) Math.primitives p entry=some final ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) → ∀ i : Lane p,Grid.active p w i →
        final.mem outRegion (Grid.offset p w i)=
          .mk .int (totalOutput (Grid.parameters p w) (Grid.atProgram entry w) i)) ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) →
        final.mem scaleRegion w.index=.mk .real (totalScale (Grid.parameters p w) (Grid.atProgram entry w))) ∧
      RawKernel.Frame p entry final := by
  classical
  obtain ⟨final,success,values,scales,frame⟩ := RawKernel.complete (Math.algebra rest) Math.primitives p positive block entry
  refine ⟨final,success,?_,?_,frame⟩
  · intro w member i active
    rw [values w member i active]
    by_cases finite : Math.FiniteDomain (Grid.parameters p w) (Grid.atProgram entry w)
    · have value := Math.real_readback rest (Grid.parameters p w) positive (Grid.atProgram entry w) factor finite i active
      rw [source_readback (Math.algebra rest) Math.primitives (Grid.parameters p w) positive (Grid.atProgram entry w) i active] at value
      simpa only [totalOutput,if_pos finite] using value
    · rw [Math.quantized_invalid rest (Grid.parameters p w) positive (Grid.atProgram entry w) factor finite]
      simp only [totalOutput,if_neg finite]
  · intro w member
    rw [scales w member]
    by_cases finite : Math.FiniteDomain (Grid.parameters p w) (Grid.atProgram entry w)
    · have value := Math.real_scale rest (Grid.parameters p w) positive (Grid.atProgram entry w) factor finite
      rw [source_scale] at value
      simpa only [totalScale,if_pos finite] using value
    · rw [Math.scale_invalid rest (Grid.parameters p w) positive (Grid.atProgram entry w) factor finite]
      simp only [totalScale,if_neg finite]

/-- Every logical coordinate occurs in the full raw launch. -/
theorem all_coordinates_raw (p : Parameters) (block : 0<p.blockRows)
    (blocks : p.blocks=Geometry.blocks p.length p.blockRows) (cover : p.blockRows*p.channels ≤ p.width)
    (batch row channel : Nat) (hb : batch<p.batches) (hr : row<p.length) (hc : channel<p.channels) :
    ∃ w∈Schedule.programs 0 (p.batches*p.blocks),∃ i : Lane p,Grid.active p w i ∧
      Grid.offset p w i=batch*p.length*p.channels+row*p.channels+channel :=
  Grid.coordinate_covered p block (by omega) blocks cover batch row channel hb hr hc

#axiomsClean per_block_int8_real_raw
#axiomsClean all_coordinates_raw
#specNonCircular totalOutput avoiding [source,FP.Structural.exec,Grid.model,quantized]
#specNonCircular totalScale avoiding [source,FP.Structural.exec,Grid.model,savedScale]
end
end VeriTile.Bench.Repairs.Int8Quantization.Specification
