import VeriTile.Torch.FP32
import Mathlib.Data.Nat.Log
import Mathlib.Data.Nat.Sqrt
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaxReduction
open VeriTile.Torch

def blockChoices : List Nat := [8,16,32]
def numWarps : Nat := 8

/-- Exact integer next-power-of-two used by Triton. -/
def nextPower (n : Nat) : Nat := if n≤1 then 1 else 2^((n-1).log2+1)
def globalBlock (n : Nat) : Nat := nextPower (Nat.sqrt (n-1)+1)
def globalMid (n : Nat) : Nat := (n+globalBlock n-1)/globalBlock n
def globalWidth (n : Nat) : Nat := nextPower (globalMid n)

def TensorValid (x : TensorMetadata) (contiguous : Bool) : Prop :=
  x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" ∧
  x.isNeg=false ∧ (contiguous=true → x.isContiguous=true) ∧ x.numel<2^31 ∧
  (0<x.numel → x.dataPtr.toNat%4=0)

def GlobalContract (x : TensorMetadata) : Prop := TensorValid x true ∧ 0<x.numel

structure DimArguments where
  input : TensorMetadata
  /-- The result of Python's operator.index classification. -/
  dim : Int
  /-- The truth-value observation used by the original keepdim branch. -/
  keepdim : Bool

def DimArguments.axis (a : DimArguments) : Nat := (a.dim % (a.input.shape.length : Int)).toNat
def DimArguments.M (a : DimArguments) : Nat := (a.input.shape.take a.axis).prod
def DimArguments.N (a : DimArguments) : Nat := a.input.shape.getD a.axis 0
def DimArguments.K (a : DimArguments) : Nat := (a.input.shape.drop (a.axis+1)).prod
def DimArguments.outputShape (a : DimArguments) : List Nat :=
  a.input.shape.take a.axis ++ (if a.keepdim then [1] else []) ++ a.input.shape.drop (a.axis+1)

def DimContract (a : DimArguments) : Prop :=
  TensorValid a.input false ∧
  -(a.input.shape.length : Int)≤a.dim ∧ a.dim<(a.input.shape.length : Int) ∧
  0<a.N ∧ a.N≤32768 ∧ (a.M=0 ∨ a.K≤65535)

inductive InputError where
  | tensor | metadata | empty | dimension | width | grid
  deriving DecidableEq, Repr

noncomputable def checkGlobal (x : ArgumentMetadata) : Except InputError Unit := by
  classical
  exact match x with
  | .nonTensor => .error .tensor
  | .tensor t => if ¬TensorValid t true then .error .metadata
    else if t.numel=0 then .error .empty else .ok ()

noncomputable def checkDimension (a : DimArguments) : Except InputError Unit := by
  classical
  exact if ¬TensorValid a.input false then .error .metadata
    else if ¬(-(a.input.shape.length : Int)≤a.dim ∧ a.dim<(a.input.shape.length : Int)) then .error .dimension
    else if ¬(0<a.N ∧ a.N≤32768) then .error .width
    else if ¬(a.M=0 ∨ a.K≤65535) then .error .grid else .ok ()

end VeriTile.Bench.Repairs.MaxReduction
