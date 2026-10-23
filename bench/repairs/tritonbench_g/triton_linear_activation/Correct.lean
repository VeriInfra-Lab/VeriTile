import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Correct
open VeriTile.Triton FP.Structural KernelSource KernelContract
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
set_option autoImplicit false

/-- Complete typed operand/output windows, source identity, masks and storage
precision are public in IO.lean. The saved output is active only when requested. -/
def linearIO := LinearIO.logicalRealIO

/-- Independent dense sum over every logical K coordinate, plus optional bias. -/
def preactivation (p : Parameters) (xs : LinearIO.LogicalInputs p)
    (i : Fin p.blockRows) (j : Fin p.blockColumns) : ℝ :=
  (if p.hasBias then xs.bias j else 0) + ∑ k : Fin p.inner,xs.left i k * xs.right j k

/-- All activation branches of the independent mathematical specification. -/
def activation (name : Option String) (x : ℝ) : ℝ :=
  if name=some "tanh" then Real.tanh x
  else if name=some "gelu" then x/2*(1+VeriTile.Math.realErf (x/Real.sqrt 2))
  else if name=some "fast_gelu" then
    x/2*(1+Real.tanh (Real.sqrt (2/Real.pi)*(x+0.044715*x^3)))
  else if name=some "relu" then max 0 x
  else x

/-- Named result and optional saved preactivation; the original output mask
keeps the latter active only when requested. -/
def formula (p : Parameters) (xs : LinearIO.LogicalInputs p) : LinearIO.LogicalOutputs p :=
  ⟨fun i j => activation p.activation (preactivation p xs i j),preactivation p xs⟩

/-- Every active output of the complete source, actual access safety, successful
execution and the full exterior frame. Both FP32 and AMP FP16 source modes and
all existing runtime geometry/stride choices remain explicit. -/
specification kernel_correct (mode : Mode) (p : Parameters) (pid : Nat)
    (positive : 0 < p.blockInner) (strides : CanonicalStrides p) (separate : p.saved ≠ p.output) :
    Spec.Real (linearIO mode p pid ⊨ formula p) := by
  have formula_eq : formula p = LinearIO.logicalFormula p := rfl
  rw [formula_eq]
  exact (LinearIO.logical_iff mode p pid).2
    (LinearIO.real_implements mode p pid positive strides separate)

#stmtSurfaceSubset kernel_correct ⊆
  [Spec.Real, linearIO, formula, LogicalProgramIO.Implements, Mode, Parameters,LinearIO.LogicalInputs,LinearIO.LogicalOutputs,
   Parameters.blockInner, Parameters.saved, Parameters.output, CanonicalStrides, RegionName]
#specNonCircular formula avoiding [KernelSource.source, FP.Structural.exec]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Correct
