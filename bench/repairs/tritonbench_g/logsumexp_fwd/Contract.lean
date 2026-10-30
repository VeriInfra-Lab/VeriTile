import bench.repairs.tritonbench_g.max_reduction.Proofs.Contract

namespace VeriTile.Bench.Repairs.LogsumexpFwd
open VeriTile.Torch

inductive ScaleKind where
  | none | floating | other
  deriving DecidableEq, Repr

structure Arguments where
  input : TensorMetadata
  scale : ScaleKind
  dtype : Option DType

def Arguments.rows (a : Arguments) : Nat := a.input.shape.dropLast.prod
def Arguments.columns (a : Arguments) : Nat := a.input.shape.getLastD 0
def Arguments.block (a : Arguments) : Nat := min (MaxReduction.nextPower a.columns) 65536
def Arguments.blocks (a : Arguments) : Nat := (a.columns+a.block-1)/a.block
def Arguments.hasScale (a : Arguments) : Bool := decide (a.scale=.floating)
def Arguments.outputDType (a : Arguments) : DType := a.dtype.getD (.floating .fp32)

def InputContract (a : Arguments) : Prop :=
  a.input.layout=.strided ∧ a.input.dtype=.floating .fp32 ∧ a.input.device.kind="cuda" ∧
  a.input.shape≠[] ∧ a.input.isContiguous=Bool.true ∧ a.input.isNeg=Bool.false ∧
  a.input.numel<2^31 ∧ (0<a.input.numel → a.input.dataPtr.toNat%4=0) ∧
  a.scale≠.other ∧ (a.outputDType=.floating .fp32 ∨ a.outputDType=.other "float64") ∧ a.rows<2^31

inductive InputError where
  | tensor | metadata | rank | layout | extent | alignment | scale | dtype | outputExtent
  deriving DecidableEq, Repr

noncomputable def checkInputs (a : Arguments) : Except InputError Unit := by
  classical
  exact if ¬(a.input.layout=.strided ∧ a.input.dtype=.floating .fp32 ∧ a.input.device.kind="cuda")
      then .error .metadata
    else if a.input.shape=[] then .error .rank
    else if ¬(a.input.isContiguous=Bool.true ∧ a.input.isNeg=Bool.false) then .error .layout
    else if ¬a.input.numel<2^31 then .error .extent
    else if ¬(0<a.input.numel → a.input.dataPtr.toNat%4=0) then .error .alignment
    else if a.scale=.other then .error .scale
    else if ¬(a.outputDType=.floating .fp32 ∨ a.outputDType=.other "float64") then .error .dtype
    else if ¬a.rows<2^31 then .error .outputExtent else .ok ()

end VeriTile.Bench.Repairs.LogsumexpFwd
