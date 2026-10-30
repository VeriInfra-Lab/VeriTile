import bench.tritonbench_g.kldiv_ops.Proofs.Checked.Binding
import bench.tritonbench_g.kldiv_ops.Proofs.ForwardValues
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardValues

/-! Kernel boundary for the unchanged selected FP32 checked source. Python
admission remains exactly the selected wrapper's checks. Host allocation and
postprocessing are outside the two launched-kernel results. -/
namespace VeriTile.Bench.Repairs.KldivOps.Kernel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
open scoped BigOperators
noncomputable section
set_option autoImplicit false

/-- Independent mathematical KL summand. No special zero-target extension. -/
def loss (logTarget : Bool) (eps prediction target : ℝ) : ℝ :=
  if logTarget then Real.exp target * (target - prediction)
  else target * (Real.log (max target eps) - prediction)

/-- The launched forward kernel writes elements or row sums. All three
nonzero reduction modes have this same scratch output; Python later reduces
and normalizes it. -/
def forwardTarget (logTarget : Bool) (eps : ℝ) (red : Host.Reduction) (cols : Nat)
    (prediction target : Nat → ℝ) (i : Nat) : ℝ :=
  if red = .none then loss logTarget eps (prediction i) (target i)
  else ∑ c : Fin cols, loss logTarget eps (prediction (i*cols+c.val)) (target (i*cols+c.val))

/-- Unscaled prediction derivative written by the standalone backward kernel.
The cotangent is consumed only by the later Python operation. -/
def backwardTarget (logTarget : Bool) (target : ℝ) : ℝ :=
  if logTarget then -Real.exp target else -target

/-- Selected checks plus ordinary bindings of actual Tensor storage. The output
facts are provided by the existing fresh FP32 allocation, not new user guards.
Prediction and target are allowed to alias each other. -/
structure ForwardDomain (predictionTensor targetTensor : Tensor)
    (prediction target output : View) (rows cols : Nat) (red : Host.Reduction) : Prop where
  admitted : Contiguous.ForwardContract predictionTensor targetTensor Bool.true
  predictionBinding : Binding predictionTensor prediction
  targetBinding : Binding targetTensor target
  shape : prediction.shape = [rows,cols]
  outputShape : output.shape = red.outputShape rows cols
  outputDense : Dense output
  outputPositive : output.negative = Bool.false
  outputBacked : output.backed
  outputAligned : output.storage.baseByte % 4 = 0
  predictionSeparate : prediction.storage.separate output.storage
  targetSeparate : target.storage.separate output.storage

/-- Exactly the selected backward wrapper admission, including the cotangent
shape/device/storage guard. The kernel itself reads only target. -/
structure BackwardDomain (destinationTensor targetTensor gradientTensor : Tensor)
    (target destination : View) (rows cols : Nat) : Prop where
  admitted : Contiguous.BackwardContract destinationTensor targetTensor gradientTensor
  destinationBinding : Binding destinationTensor destination
  targetBinding : Binding targetTensor target
  shape : destination.shape = [rows,cols]

/-- Ordinary representability of a successfully dispatched row grid. This is
not an added Python size guard and says nothing about source execution. -/
def Dispatch (rows : Nat) : Prop := rows ≤ 2^31

/-- Native four-byte pointers are tied to the same keys used by logical loads,
source execution and physical stores. -/
def TensorAddresses (t : Tensor) (view : View) : Prop :=
  ∀ i, i < view.numel →
    (t.metadata.dataPtr + BitVec.ofNat 64 i * BitVec.ofNat 64 4).toNat = (view.key i).address ∧
    (view.key i).address % 4 = 0 ∧
    ∀ byte, byte < 4 → (view.key i).address+byte < view.storage.baseByte+view.storage.bytes ∧
      (view.key i).address+byte < 2^64

def OutputAddresses (view : View) : Prop :=
  ∀ i, i < view.numel →
    (BitVec.ofNat 64 (view.storage.baseByte+4*view.storageOffset) +
      BitVec.ofNat 64 i * BitVec.ofNat 64 4).toNat = (view.key i).address ∧
    (view.key i).address % 4 = 0 ∧
    ∀ byte, byte < 4 → (view.key i).address+byte < view.storage.baseByte+view.storage.bytes ∧
      (view.key i).address+byte < 2^64

abbrev python_forward_checks_iff_preconditions := Contiguous.python_forward_checks_iff
abbrev python_backward_checks_iff_preconditions := Contiguous.python_backward_checks_iff

#specNonCircular loss avoiding [KldivOps.Tile.scalar]
#specNonCircular forwardTarget avoiding [KldivOps.Program.grid,KldivOps.Tile.scalar,KldivOps.Tile.chunkSum]
#specNonCircular backwardTarget avoiding [KldivOps.Backward.grid,KldivOps.Backward.scalar]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Kernel
