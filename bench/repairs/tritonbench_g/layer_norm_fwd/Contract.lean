import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPrelude

namespace VeriTile.Bench.Repairs.LayerNormFwd.Admission
open Wrapper

/-- Dynamic Python type/layout attributes, and the same tensor handle used by
both the guard and source wrapper. Invalid objects carry unused placeholder
metadata and are rejected before any attributes are read. -/
structure Input where
  value : Tensor
  isTensor : Bool
  strided : Bool
  cuda : Bool
  storageOffset : Nat
  dataPtr : Nat

inductive DTypeArgument where
  | none | other | dtype (value : DType)

def DTypeArgument.valid : DTypeArgument → Bool
  | .other => Bool.false
  | _ => Bool.true

def DTypeArgument.option : DTypeArgument → Option DType
  | .dtype value => some value
  | _ => Option.none

def InputContract (input : Input) (shape : List Nat) (strides : List Int) (device : Nat) : Prop :=
  input.isTensor = Bool.true ∧ input.strided = Bool.true ∧ input.cuda = Bool.true ∧
  input.value.dtype = DType.fp32 ∧ input.value.device = device ∧
  input.value.shape = shape ∧ input.value.strides = strides ∧
  input.storageOffset = 0 ∧ input.value.negative = Bool.false ∧
  (0 < shape.prod → input.dataPtr % 4 = 0)

def inputCheck (input : Input) (shape : List Nat) (strides : List Int) (device : Nat) : Bool :=
  input.isTensor && input.strided && input.cuda && (input.value.dtype == DType.fp32) &&
  (input.value.device == device) && (input.value.shape == shape) && (input.value.strides == strides) &&
  (input.storageOffset == 0) && !input.value.negative && (!(shape.prod > 0) || input.dataPtr % 4 == 0)

def OptionalContract (input : Option Input) (shape : List Nat) (strides : List Int) (device : Nat) : Prop :=
  ∀ value, input = some value → InputContract value shape strides device

def optionalCheck (input : Option Input) (shape : List Nat) (strides : List Int) (device : Nat) : Bool :=
  match input with
  | none => Bool.true
  | some value => inputCheck value shape strides device

/-- Raw binary64 scalar values are retained. None means a non-builtin-float
Python argument. Native literal lowering receives these exact Float payloads. -/
def EpsilonContract (value : Option Float) : Prop :=
  ∃ q, value = some q ∧ q.isNaN = Bool.false ∧ q.isInf = Bool.false ∧ 0.0 < q

def epsilonCheck (value : Option Float) : Bool :=
  match value with
  | none => Bool.false
  | some q => !q.isNaN && !q.isInf && decide (0.0 < q)

def ProbabilityContract (value : Option Float) : Prop :=
  ∃ q, value = some q ∧ q.isNaN = Bool.false ∧ q.isInf = Bool.false ∧ 0.0 ≤ q ∧ q < 1.0

def probabilityCheck (value : Option Float) : Bool :=
  match value with
  | none => Bool.false
  | some q => !q.isNaN && !q.isInf && decide (0.0 ≤ q) && decide (q < 1.0)

structure Call where
  (x weight : Input)
  (bias residual x1 weight1 bias1 rowscale : Option Input)
  (rows columns : Nat)
  (epsilon dropout : Option Float)
  (rms returnMask : Option Bool)
  (outDtype residualDtype : DTypeArgument)

def matrixStrides (c : Call) : List Int := [max c.columns 1,1]

def InputDomain (c : Call) : Prop :=
  InputContract c.x [c.rows,c.columns] (matrixStrides c) c.x.value.device ∧
  InputContract c.weight [c.columns] [1] c.x.value.device ∧
  OptionalContract c.bias [c.columns] [1] c.x.value.device ∧
  OptionalContract c.weight1 [c.columns] [1] c.x.value.device ∧
  OptionalContract c.bias1 [c.columns] [1] c.x.value.device ∧
  OptionalContract c.residual [c.rows,c.columns] (matrixStrides c) c.x.value.device ∧
  OptionalContract c.x1 [c.rows,c.columns] (matrixStrides c) c.x.value.device ∧
  OptionalContract c.rowscale [c.rows] [1] c.x.value.device ∧
  (c.x1 = none ∨ c.rowscale = none) ∧
  EpsilonContract c.epsilon ∧ ProbabilityContract c.dropout ∧
  c.rms.isSome = Bool.true ∧ c.returnMask.isSome = Bool.true ∧
  c.outDtype.valid = Bool.true ∧ (c.residual = none → c.residualDtype.valid = Bool.true)

def checkInputs (c : Call) : Bool :=
  inputCheck c.x [c.rows,c.columns] (matrixStrides c) c.x.value.device &&
  inputCheck c.weight [c.columns] [1] c.x.value.device &&
  optionalCheck c.bias [c.columns] [1] c.x.value.device &&
  optionalCheck c.weight1 [c.columns] [1] c.x.value.device &&
  optionalCheck c.bias1 [c.columns] [1] c.x.value.device &&
  optionalCheck c.residual [c.rows,c.columns] (matrixStrides c) c.x.value.device &&
  optionalCheck c.x1 [c.rows,c.columns] (matrixStrides c) c.x.value.device &&
  optionalCheck c.rowscale [c.rows] [1] c.x.value.device &&
  (!c.x1.isSome || !c.rowscale.isSome) &&
  epsilonCheck c.epsilon && probabilityCheck c.dropout &&
  c.rms.isSome && c.returnMask.isSome && c.outDtype.valid &&
  (c.residual.isSome || c.residualDtype.valid)

def request (c : Call) : Request where
  x := c.x.value
  weight := c.weight.value
  bias := c.bias.map Input.value
  residual := c.residual.map Input.value
  x1 := c.x1.map Input.value
  weight1 := c.weight1.map Input.value
  bias1 := c.bias1.map Input.value
  rowscale := c.rowscale.map Input.value
  outDtype := c.outDtype.option
  residualDtype := c.residualDtype.option
  eps := c.epsilon.getD 0.0
  dropoutP := c.dropout.getD 0.0
  rms := c.rms.getD Bool.false
  returnMask := c.returnMask.getD Bool.false

end VeriTile.Bench.Repairs.LayerNormFwd.Admission
