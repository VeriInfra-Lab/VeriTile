import bench.tritonbench_g.fused_activation.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.FusedActivation.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- The independent scalar formula is (input/2 + x + bias), followed by the
selected sigmoid/ReLU. RealIO.lean contains its definition and visible IO fields. -/
def formula := RealIO.formula

/-- Three entry-state reads, cyclic bias indexing, and an in-place x write.
The complete public channel/window/mask data is shared in RealIO.lean. -/
def activationIO := RealIO.logicalActivationIO

/-- Complete regional kernel IO, including actual source access safety,
independent mathematical output values and the exact exterior frame. -/
specification fused_activation_kernel_correctness (x b input : RegionName)
    (W N B : Nat) (a : Bool) :
    Spec.Real (activationIO x b input W N B a ⊨ fun xs j =>
      let z := (1/2) * xs.input j + xs.x j + xs.bias j
      if a then Real.sigmoid z else max 0 z) :=
  RealInterface.logical_correct x b input W N B a

#stmtSurfaceSubset fused_activation_kernel_correctness ⊆
  [Spec.Real, activationIO, formula, RegionName, LogicalIO.Implements,
   RealIO.ActivationValues, RealIO.ActivationValues.x, RealIO.ActivationValues.bias,
   RealIO.ActivationValues.input, RealIO.activationIO, GroupedMasked2DKernelIO.RegionalImplements,
   GroupedMasked2DKernelIO.nIn, GroupedMasked2DKernelIO.B]
#specNonCircular formula avoiding [kernel, commutedKernel]
#axiomsClean fused_activation_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedActivation.Correct
