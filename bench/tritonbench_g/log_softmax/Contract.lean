import bench.tritonbench_g.max_reduction.Proofs.Contract

namespace VeriTile.Bench.Repairs.LogSoftmax
open VeriTile.Torch

structure Arguments where
  input : TensorMetadata
  /-- Result of operator.index; unsuccessful classification raises TypeError. -/
  dim : Int
  dtype : Option DType

def Arguments.axis (a : Arguments) : Nat := (a.dim%(a.input.shape.length : Int)).toNat
def Arguments.M (a : Arguments) : Nat := (a.input.shape.take a.axis).prod
def Arguments.N (a : Arguments) : Nat := a.input.shape.getD a.axis 0
def Arguments.K (a : Arguments) : Nat := (a.input.shape.drop (a.axis+1)).prod

def ShapeContract (a : Arguments) : Prop :=
  -(a.input.shape.length : Int)≤a.dim ∧ a.dim<(a.input.shape.length : Int) ∧
  (a.input.numel=0 ∨ a.N≤131072 ∧ a.K≤65535)

def InputContract (a : Arguments) : Prop :=
  MaxReduction.TensorValid a.input Bool.false ∧
  (a.dtype=none ∨ a.dtype=some (.floating .fp32)) ∧ ShapeContract a

structure BackwardArguments where
  saved : Arguments
  gradient : TensorMetadata

def BackwardContract (a : BackwardArguments) : Prop :=
  MaxReduction.TensorValid a.saved.input Bool.true ∧
  MaxReduction.TensorValid a.gradient Bool.false ∧
  a.gradient.shape=a.saved.input.shape ∧ a.gradient.device=a.saved.input.device ∧ ShapeContract a.saved

inductive InputError where
  | tensor | metadata | dtype | dimension | extent | gradient | savedLayout
  deriving DecidableEq, Repr

noncomputable def checkInputs (a : Arguments) : Except InputError Unit := by
  classical
  exact if ¬MaxReduction.TensorValid a.input Bool.false then .error .metadata
    else if ¬(a.dtype=none ∨ a.dtype=some (.floating .fp32)) then .error .dtype
    else if ¬(-(a.input.shape.length : Int)≤a.dim ∧ a.dim<(a.input.shape.length : Int)) then .error .dimension
    else if ¬(a.input.numel=0 ∨ a.N≤131072 ∧ a.K≤65535) then .error .extent else .ok ()

noncomputable def checkBackward (a : BackwardArguments) : Except InputError Unit := by
  classical
  exact if ¬MaxReduction.TensorValid a.saved.input Bool.false then .error .metadata
    else if ¬MaxReduction.TensorValid a.gradient Bool.false then .error .metadata
    else if ¬(a.gradient.shape=a.saved.input.shape ∧ a.gradient.device=a.saved.input.device) then .error .gradient
    else if a.saved.input.isContiguous≠Bool.true then .error .savedLayout
    else if ¬ShapeContract a.saved then .error .dimension else .ok ()

end VeriTile.Bench.Repairs.LogSoftmax
