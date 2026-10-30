import VeriTile.Torch.Tensor
import bench.tritonbench_g.fused_layernorm_triton.Kernels

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Contract

structure Metadata extends VeriTile.Torch.TensorMetadata where
  storageOffset : Nat
  deriving DecidableEq, Repr

inductive Argument where
  | nonTensor
  | tensor (metadata : Metadata)
  deriving DecidableEq, Repr

def rows (x : Metadata) : Nat := x.shape.getD 0 0
def width (x : Metadata) : Nat := x.shape.getD 1 0

structure TensorContract (m : Metadata) : Prop where
  layout : m.layout = .strided
  dtype : m.dtype = .floating .fp32
  cuda : m.device.kind = "cuda"
  device : m.device.index = 0
  contiguous : m.isContiguous = Bool.true
  offset : m.storageOffset = 0
  noNeg : m.isNeg = Bool.false
  alignment : 0 < m.numel → m.dataPtr.toNat % 4 = 0

structure InputContract (w b x : Metadata) : Prop where
  weight : TensorContract w
  bias : TensorContract b
  input : TensorContract x
  rank : x.shape.length = 2
  positive : 0 < width x
  weightShape : w.shape = [width x]
  biasShape : b.shape = [width x]

inductive Error where
  | tensor | type | device | contiguous | negative | alignment | shape
  deriving DecidableEq, Repr

def checkTensor (m : Metadata) : Except Error Unit :=
  if m.layout != .strided || m.dtype != .floating .fp32 then .error .type
  else if m.device.kind != "cuda" || m.device.index != 0 then .error .device
  else if !m.isContiguous || m.storageOffset != 0 then .error .contiguous
  else if m.isNeg then .error .negative
  else if m.numel > 0 && m.dataPtr.toNat % 4 != 0 then .error .alignment
  else .ok ()

def checkInputs (w b x : Metadata) : Except Error Unit :=
  match checkTensor w, checkTensor b, checkTensor x with
  | .error e, _, _ => .error e
  | _, .error e, _ => .error e
  | _, _, .error e => .error e
  | .ok _, .ok _, .ok _ =>
    if x.shape.length != 2 || width x == 0 then .error .shape
    else if w.shape != [width x] || b.shape != [width x] then .error .shape
    else .ok ()

def checkArguments (w b x : Argument) : Except Error Unit :=
  match w,b,x with
  | .tensor w, .tensor b, .tensor x => checkInputs w b x
  | _,_,_ => .error .tensor

def AdmittedArguments (w b x : Argument) : Prop :=
  match w,b,x with
  | .tensor w, .tensor b, .tensor x => InputContract w b x
  | _,_,_ => False

open VeriTile.Triton
noncomputable section

/-- Both original autotune configurations; compiler tree order is explicit. -/
structure Configuration where
  block : Nat
  allowed : block = 1024 ∨ block = 2048
  tree : Kernels.ReductionTree block

theorem Configuration.positive (c : Configuration) : 0 < c.block := by
  rcases c.allowed with h | h <;> omega

def source (X W Bias Y Mean Rstd : RegionName) (S D : Nat)
    (c : Configuration) : ComputeKernel :=
  Kernels.kernel X W Bias Y Mean Rstd S D c.block c.tree

/-- Mathematical layer normalization over the logical columns. -/
def mean (D : Nat) (x : Nat → ℝ) : ℝ := (∑ j ∈ Finset.range D, x j) / D
def variance (D : Nat) (x : Nat → ℝ) : ℝ :=
  (∑ j ∈ Finset.range D, (x j - mean D x)^2) / D
def output (D : Nat) (x w bias : Nat → ℝ) (j : Nat) : ℝ :=
  (x j - mean D x) * (1 / Real.sqrt (variance D x + 1e-5)) * w j + bias j


end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Contract
