import bench.tritonbench_g.quant_transpose_kernel.Kernels

namespace VeriTile.Bench.Repairs.QuantTransposeKernel
open QuantTransposeFinite

/-- Source-int32 envelopes, including padded inactive lanes. -/
structure IndexDomain (a : Input) : Prop where
  rows : 0 < a.M
  columns : 0 < a.N
  rowsUpper : a.M+128 < 2^31
  columnsUpper : a.N+128 < 2^31
  strideM : a.sam < 2^31
  strideN : a.san < 2^31
  grid : gridM a.M*gridN a.N < 2^31
  width : 8*gridN a.N < 2^31
  inputOffset : (gridM a.M*128)*a.sam+(gridN a.N*128)*a.san < 2^31
  outputOffset : gridM a.M*128+(gridN a.N*128)*a.M < 2^31

structure Metadata where
  input : Input
  fp32 : Bool
  resolved : Bool
  currentDevice : Nat
  dataPointer : Nat

/-- The tensor/layout/rank and nonnegative-stride classifications are checked
before constructing this exact ordinary Tensor descriptor. -/
def metadataChecks (p : Metadata) : Bool :=
  decide (p.fp32 = Bool.true ∧ p.resolved = Bool.true ∧
    p.input.isCuda = Bool.true ∧ p.input.deviceIndex = p.currentDevice ∧
    p.input.shape.length = 2 ∧ (p.input.sam = 1 ∨ p.input.san = 1) ∧ p.dataPointer%4=0)

def indexChecks (a : Input) : Bool :=
  decide (0 < a.M ∧ 0 < a.N ∧ a.M+128 < 2^31 ∧ a.N+128 < 2^31 ∧
    a.sam < 2^31 ∧ a.san < 2^31 ∧ gridM a.M*gridN a.N < 2^31 ∧
    8*gridN a.N < 2^31 ∧
    (gridM a.M*128)*a.sam+(gridN a.N*128)*a.san < 2^31 ∧
    gridM a.M*128+(gridN a.N*128)*a.M < 2^31)

structure InputContract (p : Metadata) : Prop where
  metadata : metadataChecks p = Bool.true
  indices : IndexDomain p.input

inductive InputError where
  | tensor | layout | rank | negativeStride | metadata | indices
  deriving DecidableEq, Repr

def checkInputs (p : Metadata) : Except InputError Unit :=
  if !metadataChecks p then .error .metadata
  else if !indexChecks p.input then .error .indices
  else .ok ()

inductive Arguments where
  | nonTensor | nonStrided | nonMatrix | negativeStride
  | tensor (metadata : Metadata)

def checkArguments (args : Arguments) : Except InputError Unit :=
  match args with
  | .nonTensor => .error .tensor
  | .nonStrided => .error .layout
  | .nonMatrix => .error .rank
  | .negativeStride => .error .negativeStride
  | .tensor p => checkInputs p

def Admitted (args : Arguments) : Prop :=
  match args with
  | .tensor p => InputContract p
  | _ => False

end VeriTile.Bench.Repairs.QuantTransposeKernel
