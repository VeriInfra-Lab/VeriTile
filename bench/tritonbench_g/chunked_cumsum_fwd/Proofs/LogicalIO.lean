import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.IO
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.IOReal

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.LogicalTensors
open VeriTile Triton FP.Structural Kernel Kernel.Scalar
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete original judgment, in both directions, for the original
canonical formula after bijective input/output tensor coordinates. -/
theorem full_canonical_iff (c : Geometry) (plan : PrefixPlan c.BH c.BC) (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    (io c plan hasBias reverseBias useSoftplus lower upper ⊨ fun values => outputs c ((fun values => Interface.result c (RealTarget.prepared c (Interface.logicalInputs c values) hasBias useSoftplus lower upper) (RealTarget.cumulative c (Interface.logicalInputs c values) hasBias useSoftplus lower upper)) ((inputs c).symm values))) ↔
      (Interface.programIO (RealTarget.model reverseBias upper) c plan hasBias useSoftplus lower (RealTarget.upperWord upper)).Implements (fun values => Interface.result c (RealTarget.prepared c (Interface.logicalInputs c values) hasBias useSoftplus lower upper) (RealTarget.cumulative c (Interface.logicalInputs c values) hasBias useSoftplus lower upper)) := by
  change (Interface.programIO (RealTarget.model reverseBias upper) c plan hasBias useSoftplus lower (RealTarget.upperWord upper)).Implements (fun raw => (outputs c).symm (outputs c ((fun values => Interface.result c (RealTarget.prepared c (Interface.logicalInputs c values) hasBias useSoftplus lower upper) (RealTarget.cumulative c (Interface.logicalInputs c values) hasBias useSoftplus lower upper)) ((inputs c).symm (inputs c raw))))) ↔ (Interface.programIO (RealTarget.model reverseBias upper) c plan hasBias useSoftplus lower (RealTarget.upperWord upper)).Implements (fun values => Interface.result c (RealTarget.prepared c (Interface.logicalInputs c values) hasBias useSoftplus lower upper) (RealTarget.cumulative c (Interface.logicalInputs c values) hasBias useSoftplus lower upper))
  simp only [Equiv.symm_apply_apply]


theorem correct (c : Geometry) (plan : PrefixPlan c.BH c.BC) (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (sizes : Sizes c) :
    io c plan hasBias reverseBias useSoftplus lower upper ⊨ fun values => outputs c ((fun values => Interface.result c (RealTarget.prepared c (Interface.logicalInputs c values) hasBias useSoftplus lower upper) (RealTarget.cumulative c (Interface.logicalInputs c values) hasBias useSoftplus lower upper)) ((inputs c).symm values)) :=
  (full_canonical_iff c plan hasBias reverseBias useSoftplus lower upper).mpr (Interface.real_implements c sizes plan hasBias reverseBias useSoftplus lower upper)

#axiomsClean full_canonical_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.LogicalTensors
