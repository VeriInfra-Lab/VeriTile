import bench.repairs.tritonbench_g.l2_norm_triton1.Contract
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.BigOperators.Fin

namespace VeriTile.Bench.Repairs.L2NormBwd
abbrev TensorMetadata := L2NormTriton1.TensorMetadata
abbrev ArgumentMetadata := L2NormTriton1.ArgumentMetadata

/-- Direct views retain DY grouping and its possibly longer logical prefix. -/
structure InputContract (x dy : TensorMetadata) : Prop where
  layout : x.layout=.strided ∧ dy.layout=.strided
  dtype : x.dtype=.floating .fp32 ∧ dy.dtype=.floating .fp32
  cuda : x.device.kind="cuda" ∧ dy.device.kind="cuda"
  sameDevice : x.device=dy.device
  rank : dy.shape≠[]
  width : 0<dy.shape.getLastD 0 ∧ dy.shape.getLastD 0≤16384
  divisible : x.numel%dy.shape.getLastD 0=0
  enough : x.numel≤dy.numel
  contiguous : x.isContiguous=Bool.true ∧ dy.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false ∧ dy.isNeg=Bool.false
  alignment : 0<x.numel → x.dataPtr.toNat%4=0 ∧ dy.dataPtr.toNat%4=0

inductive InputError where
  | tensors | layout | dtype | cuda | device | rank | width | divisible | short | contiguous | negative | alignment
  deriving DecidableEq, Repr

def checkTensorInputs (x dy : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || dy.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || dy.dtype != .floating .fp32 then .error .dtype
  else if !(x.isCuda && dy.isCuda) then .error .cuda
  else if x.device != dy.device then .error .device
  else if dy.shape=[] then .error .rank
  else if dy.shape.getLastD 0=0 || dy.shape.getLastD 0>16384 then .error .width
  else if x.numel%dy.shape.getLastD 0 != 0 then .error .divisible
  else if dy.numel<x.numel then .error .short
  else if !(x.isContiguous && dy.isContiguous) then .error .contiguous
  else if x.isNeg || dy.isNeg then .error .negative
  else if x.numel>0 && (x.dataPtr.toNat%4 != 0 || dy.dataPtr.toNat%4 != 0) then .error .alignment
  else .ok ()

def checkInputs (x dy : ArgumentMetadata) : Except InputError Unit :=
  match x,dy with
  | .tensor x,.tensor dy => checkTensorInputs x dy
  | _,_ => .error .tensors

def AdmittedArguments (x dy : ArgumentMetadata) : Prop :=
  match x,dy with
  | .tensor x,.tensor dy => InputContract x dy
  | _,_ => False
namespace Math
noncomputable section
open scoped BigOperators

/-- Independent logical derivative formula over exactly N entries. Its row
domain is i<N at the caller boundary. Total sqrt/div conventions extend it
outside q>0; no result is read from execution and no FP=Real law is assumed. -/
def derivative (N : Nat) (eps : ℝ) (xs dys : Nat → ℝ) (i : Nat) : ℝ :=
  let q := (∑ j : Fin N,xs j.val*xs j.val)+eps
  let dot := ∑ j : Fin N,dys j.val*xs j.val
  dys i*(1/Real.sqrt q)-dot*(1/q)*(1/Real.sqrt q)*xs i

end
end Math

end VeriTile.Bench.Repairs.L2NormBwd
