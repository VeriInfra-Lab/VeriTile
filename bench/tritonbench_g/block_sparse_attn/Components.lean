import bench.tritonbench_g.block_sparse_attn.Proofs.Reference

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/- Named projections of the independent recurrence. These definitions only
factor its scalar expressions; their equality with `update` is definitional. -/
namespace Component

def scores (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) : Matrix α := fun i o =>
  let raw := scoreAccumulation a native p s column p.blocks i o
  let mask := if column*16+i.2.1.val≤(row s i.1:Int) then a.literal 0 else a.negInf
  if native then a.fma (a.argument p.scale) raw mask
  else a.add (a.mul raw (a.argument p.scale)) mask

def maximum (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) : Rows α := fun i =>
  reduceLine a a.max (fun j => scores a native p s column (i,j,PUnit.unit))

def probability (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) : Matrix α := fun i o =>
  a.exp (a.sub (scores a native p s column i o) (maximum a native p s column i.1 o))

def denominator (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) : Rows α := fun i =>
  reduceLine a a.add (fun j => probability a native p s column (i,j,PUnit.unit))

def nextMaximum (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Rows α := fun i o =>
  a.max (old.maximum i o) (maximum a native p s column i o)

def alpha (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Rows α := fun i o =>
  a.exp (a.sub (old.maximum i o) (nextMaximum a native p s column old i o))

def beta (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Rows α := fun i o =>
  a.exp (a.sub (maximum a native p s column i o) (nextMaximum a native p s column old i o))

def nextDenominator (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Rows α := fun i o =>
  if native then a.fma (old.denominator i o) (alpha a native p s column old i o)
    (a.mul (denominator a native p s column i o) (beta a native p s column old i o))
  else a.add (a.mul (alpha a native p s column old i o) (old.denominator i o))
    (a.mul (beta a native p s column old i o) (denominator a native p s column i o))

def scaledProbability (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Matrix α := fun i o =>
  a.mul (probability a native p s column i o)
    (a.div (beta a native p s column old i.1 o) (nextDenominator a native p s column old i.1 o))

def scale (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Rows α := fun i o =>
  if native then a.mul (alpha a native p s column old i o)
    (a.div (old.denominator i o) (nextDenominator a native p s column old i o))
  else a.mul (a.div (old.denominator i o) (nextDenominator a native p s column old i o))
    (alpha a native p s column old i o)

def scaledOutput (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Matrix α := fun i o =>
  a.mul (old.output i o) (scale a native p s column old i.1 o)

def output (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) : Matrix α :=
  if native then tensor a (batch := []) (scaledProbability a native p s column old)
    (value a p s column) (scaledOutput a native p s column old)
  else fun i => binary a .add (scaledOutput a native p s column old i)
    (tensor a (batch := []) (scaledProbability a native p s column old)
      (value a p s column) (fun _ => broadcast a.zero) i)
end Component

theorem update_components (a : Arithmetic α) (native : Bool) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (old : Carry α) :
    update a native p s column old =
      ⟨Component.nextMaximum a native p s column old,
        Component.nextDenominator a native p s column old,
        Component.output a native p s column old⟩ := rfl

#axiomsClean update_components
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
