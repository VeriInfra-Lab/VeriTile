import VeriTile.Torch.Tensor
import bench.tritonbench_g.swiglu_fwd.Proofs.WholeIndex

/-! Checked payload and prepared-launch metadata. Bounds and ownership are
checks here; ordinary Tensor allocation/element bounds are separate API laws. -/
namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole

structure PayloadContract (x : VeriTile.Torch.TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  noNeg : x.isNeg = Bool.false

def checkPayload : VeriTile.Torch.ArgumentMetadata → Bool
  | .nonTensor => Bool.false
  | .tensor x => decide (x.layout = .strided) && decide (x.dtype = .floating .fp32) &&
    x.isCuda && !x.isNeg

def ArgumentsAdmitted : VeriTile.Torch.ArgumentMetadata → Prop
  | .nonTensor => False
  | .tensor x => PayloadContract x

def inputShapeOK (shape : List Nat) : Bool :=
  !shape.isEmpty && decide (0 < shape.getLastD 0) && decide (shape.getLastD 0 % 2 = 0)

structure LaunchMetadata where
  rows : Nat
  cols : Nat
  inputStride : Nat
  outputStride : Nat
  inputDevice : Nat
  outputDevice : Nat
  inputAllocation : Nat
  outputAllocation : Nat
  inputBase : Nat
  outputBase : Nat
  inputPointer : Nat
  outputPointer : Nat

def inputEnd (m : LaunchMetadata) : Nat :=
  m.inputPointer + 4*((m.rows-1)*m.inputStride + 2*m.cols)

def outputEnd (m : LaunchMetadata) : Nat :=
  m.outputPointer + 4*((m.rows-1)*m.outputStride + m.cols)

structure LaunchContract (m : LaunchMetadata) : Prop where
  positive : 0 < m.cols
  sameDevice : m.outputDevice = m.inputDevice
  rows32 : m.rows < 2^31
  inputStride32 : m.inputStride < 2^31
  outputStride32 : m.outputStride < 2^31
  inputPadded32 : m.rows*m.inputStride + m.cols+1024 < 2^31
  outputPadded32 : m.rows*m.outputStride + m.cols+1024 < 2^31
  unique : m.rows ≤ 1 ∨ m.cols ≤ m.outputStride
  alignment : 0 < m.rows → m.inputPointer % 4 = 0 ∧ m.outputPointer % 4 = 0
  separate : 0 < m.rows →
    outputEnd m ≤ m.inputPointer ∨ inputEnd m ≤ m.outputPointer

def checkLaunch (m : LaunchMetadata) : Bool :=
  decide (0 < m.cols) && decide (m.outputDevice = m.inputDevice) &&
  decide (m.rows < 2^31) && decide (m.inputStride < 2^31) &&
  decide (m.outputStride < 2^31) &&
  decide (m.rows*m.inputStride + m.cols+1024 < 2^31) &&
  decide (m.rows*m.outputStride + m.cols+1024 < 2^31) &&
  decide (m.rows ≤ 1 ∨ m.cols ≤ m.outputStride) &&
  decide (0 < m.rows → m.inputPointer % 4 = 0 ∧ m.outputPointer % 4 = 0) &&
  decide (0 < m.rows →
    outputEnd m ≤ m.inputPointer ∨ inputEnd m ≤ m.outputPointer)

end VeriTile.Bench.Repairs.SwigluFwd
