import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.LayerNormOps.Admission

structure Metadata where
  tensor : VeriTile.Torch.TensorMetadata
  strides : List Nat
  storageOffset : Nat
  deriving DecidableEq, Repr

inductive Argument where
  | none | other | tensor (value : Metadata)

def Argument.present : Argument → Bool
  | .none => Bool.false
  | _ => Bool.true

inductive ScalarArgument where
  | other
  /-- None represents nonfinite builtin Python floats. -/
  | float (value : Option ℝ)

inductive DTypeOption where
  | none | fp32 | other
  deriving DecidableEq, Repr

def TensorContract (arg : Argument) (shape strides : List Nat) : Prop :=
  match arg with
  | .tensor m =>
    m.tensor.layout = .strided ∧ m.tensor.dtype = .floating .fp32 ∧
    m.tensor.device = ⟨"cuda", 0⟩ ∧ m.tensor.shape = shape ∧ m.strides = strides ∧
    m.storageOffset = 0 ∧ m.tensor.isNeg = Bool.false ∧
    (0 < m.tensor.numel → m.tensor.dataPtr.toNat % 4 = 0)
  | _ => False

def tensorCheck (arg : Argument) (shape strides : List Nat) : Bool :=
  match arg with
  | .tensor m =>
    m.tensor.isStrided && m.tensor.isFP32 && (m.tensor.device == ⟨"cuda", 0⟩) &&
    (m.tensor.shape == shape) && (m.strides == strides) && (m.storageOffset == 0) &&
    !m.tensor.isNeg && (!(m.tensor.numel > 0) || m.tensor.dataPtr.toNat % 4 == 0)
  | _ => Bool.false

def OptionalContract (arg : Argument) (shape strides : List Nat) : Prop :=
  match arg with
  | .none => True
  | _ => TensorContract arg shape strides

def optionalCheck (arg : Argument) (shape strides : List Nat) : Bool :=
  match arg with
  | .none => Bool.true
  | _ => tensorCheck arg shape strides

def ScalarContract (arg : ScalarArgument) : Prop :=
  ∃ q, arg = .float (some q) ∧ 0 < q

noncomputable def scalarCheck (arg : ScalarArgument) : Bool := by
  classical
  exact match arg with
  | .float (some q) => decide (0 < q)
  | _ => Bool.false

/-- On N≥1, the source helper next_power_of_2(N) is 2^ceil(log2 N). -/
def blockSize (columns : Nat) : Nat := min 16384 (2 ^ Nat.clog 2 columns)
def rowsPerProgram (rows programs : Nat) : Nat := (rows + programs - 1) / programs

structure Common where
  (x weight bias : Argument)
  (rows columns : Nat)
  epsilon : ScalarArgument
  /-- Some b means the original Python argument has exact bool type and value b. -/
  rms : Option Bool
  currentDevice : Nat

def CommonContract (c : Common) : Prop :=
  0 < c.columns ∧ c.columns ≤ 16384 ∧
  TensorContract c.x [c.rows,c.columns] [c.columns,1] ∧
  TensorContract c.weight [c.columns] [1] ∧ OptionalContract c.bias [c.columns] [1] ∧
  c.currentDevice = 0 ∧ ScalarContract c.epsilon ∧ c.rms.isSome = Bool.true ∧
  c.rows * c.columns + blockSize c.columns < 2^31

noncomputable def commonCheck (c : Common) : Bool :=
  (c.columns > 0) && (c.columns ≤ 16384) &&
  tensorCheck c.x [c.rows,c.columns] [c.columns,1] &&
  tensorCheck c.weight [c.columns] [1] && optionalCheck c.bias [c.columns] [1] &&
  (c.currentDevice == 0) && scalarCheck c.epsilon && c.rms.isSome &&
  (c.rows * c.columns + blockSize c.columns < 2^31)

structure ForwardCall where
  common : Common
  residual : Argument
  (outDType residualDType : DTypeOption)

def ForwardContract (c : ForwardCall) : Prop :=
  CommonContract c.common ∧ c.outDType ≠ .other ∧
  OptionalContract c.residual [c.common.rows,c.common.columns] [c.common.columns,1] ∧
  (c.residual.present = Bool.false → c.residualDType ≠ .other)

noncomputable def forwardCheck (c : ForwardCall) : Bool :=
  commonCheck c.common && (c.outDType != .other) &&
  optionalCheck c.residual [c.common.rows,c.common.columns] [c.common.columns,1] &&
  (c.residual.present || c.residualDType != .other)

structure BackwardCall where
  common : Common
  (dy dresidual mean rstd : Argument)
  xDType : DTypeOption
  (hasResidual recompute : Option Bool)
  /-- The actual CUDA property query, not a caller-selected launch size. -/
  programs : Nat
  /-- Actual result of the retained math.ceil(M / sm_count) source computation. -/
  rowWidth : Nat

def MeanContract (c : BackwardCall) : Prop :=
  if c.common.rms.getD Bool.false then c.mean.present = Bool.false
  else TensorContract c.mean [c.common.rows] [1]

def meanCheck (c : BackwardCall) : Bool :=
  if c.common.rms.getD Bool.false then !c.mean.present
  else tensorCheck c.mean [c.common.rows] [1]

def BackwardContract (c : BackwardCall) : Prop :=
  CommonContract c.common ∧
  TensorContract c.dy [c.common.rows,c.common.columns] [c.common.columns,1] ∧
  OptionalContract c.dresidual [c.common.rows,c.common.columns] [c.common.columns,1] ∧
  MeanContract c ∧ TensorContract c.rstd [c.common.rows] [1] ∧ c.xDType ≠ .other ∧
  c.hasResidual.isSome = Bool.true ∧ c.recompute.isSome = Bool.true ∧ 0 < c.programs ∧
  c.rowWidth = rowsPerProgram c.common.rows c.programs ∧
  c.programs * c.rowWidth * c.common.columns + blockSize c.common.columns < 2^31 ∧
  c.programs * c.common.columns + blockSize c.common.columns < 2^31

noncomputable def backwardCheck (c : BackwardCall) : Bool :=
  commonCheck c.common &&
  tensorCheck c.dy [c.common.rows,c.common.columns] [c.common.columns,1] &&
  optionalCheck c.dresidual [c.common.rows,c.common.columns] [c.common.columns,1] &&
  meanCheck c && tensorCheck c.rstd [c.common.rows] [1] && (c.xDType != .other) &&
  c.hasResidual.isSome && c.recompute.isSome && (c.programs > 0) &&
  (c.rowWidth == rowsPerProgram c.common.rows c.programs) &&
  (c.programs * c.rowWidth * c.common.columns + blockSize c.common.columns < 2^31) &&
  (c.programs * c.common.columns + blockSize c.common.columns < 2^31)

end VeriTile.Bench.Repairs.LayerNormOps.Admission
