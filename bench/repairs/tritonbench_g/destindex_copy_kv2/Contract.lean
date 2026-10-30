import bench.repairs.tritonbench_g.destindex_copy_kv2.Kernels
import bench.repairs.tritonbench_g.destindex_copy_kv1.Contract

namespace VeriTile.Bench.Repairs.DestindexCopyKv2
noncomputable section
abbrev RowCopy := @VeriTile.Bench.Repairs.DestindexCopyKv1.RowCopy
abbrev Inputs := DestindexCopyKv1.Inputs
abbrev Word := DestindexCopyKv1.Word
abbrev Memory := DestindexCopyKv1.Memory
abbrev config := DestindexCopyKv1.config
abbrev base := DestindexCopyKv1.base
abbrev addr := DestindexCopyKv1.addr
abbrev destination := DestindexCopyKv1.destination
abbrev TensorBinding := DestindexCopyKv1.TensorBinding

def Nonempty (m : Inputs) : Prop := 0<(config m).count ∧ 0<(config m).heads ∧ 0<(config m).dims
def Width (m : Inputs) : Prop := Nonempty m → block (config m).dims=(config m).dims
structure InputContract (m : Inputs) (w : Nat → Word) : Prop where
  common : DestindexCopyKv1.InputContract m w
  width : Width m
inductive InputError where
  | common (reason : DestindexCopyKv1.InputError)
  | width
open Classical in
def checkInputs (m : Inputs) (w : Nat → Word) : Except InputError Unit :=
  match DestindexCopyKv1.checkInputs m w with
  | .error reason => .error (.common reason)
  | .ok _ => if Width m then .ok () else .error .width

def checkArguments (m : Option Inputs) (w : Nat → Word) : Except InputError Unit :=
  match m with | none => .error (.common .tensors) | some m => checkInputs m w

def Admitted (m : Option Inputs) (w : Nat → Word) : Prop :=
  match m with | none => False | some m => InputContract m w
end
end VeriTile.Bench.Repairs.DestindexCopyKv2
