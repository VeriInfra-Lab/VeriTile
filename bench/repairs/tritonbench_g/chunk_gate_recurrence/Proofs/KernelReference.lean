import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelOps

/-! Input-only floating recurrences. Every source load/cast, first zero carry,
operation order and full row-major reduction input is explicit. Native scalar
graphs and a complete reduction tree will interpret these callbacks; no source
result or execution witness occurs in the reference. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReference
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
variable {α : Type} [Inhabited α]

structure Inputs (α : Type) where
  source : Nat → Lane → α
  gate : Nat → α
  seed : Lane → α
  incoming : Nat → Lane → α
  saved : Nat → Lane → α

def zero (M : Algebra α) : α := M.literal (some .fp32) .real 0
def update (M : Algebra α) (carry gate incoming : α) : α :=
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul carry (M.fp32Load gate)) (M.fp32Load incoming)
def initial (M : Algebra α) (a : Inputs α) (seed : Bool) : Lane → α := fun i =>
  if seed then M.cast (some .fp32) .real .real (a.seed i) else zero M
def forward (M : Algebra α) (a : Inputs α) (seed : Bool) : Nat → Lane → α
  | 0 => initial M a seed
  | t+1 => fun i => update M (forward M a seed t i) (a.gate t) (a.source t i)
def reverse (M : Algebra α) (a : Inputs α) (N : Nat) : Nat → Lane → α
  | 0 => fun _ => zero M
  | n+1 => fun i => update M (reverse M a N n i) (a.gate (N-1-n)) (a.incoming (N-1-n) i)
def product (M : Algebra α) (carry saved : Lane → α) : Lane → α := fun i =>
  M.binary (some .fp32) .real .mul (carry i) (M.cast (some .fp32) .fp32 .real (saved i))
def flattened (x : Lane → α) : Values α .real [64*16] :=
  fun i => x (TileShape.reshapeIndex (input:=[64,16]) (output:=[64*16]) (by decide) i)
def contraction (M : Algebra α) (carry saved : Lane → α) : α :=
  (M.reduceSum (some .fp32) (shape:=[64*16]) ⟨0,by decide⟩ Bool.false
    (flattened (product M carry saved))) PUnit.unit
def stored (M : Algebra α) (dtype : FloatDType) (x : α) : Cell α :=
  .mk dtype.toTileDType (ofFloat dtype (M.cast (some .fp32) .real dtype x))

def read (c : Shape) (s : SeedStrides) (p : Program c) (u : State α) : Inputs α where
  source := fun t => raw "S" .real (fun e j => address c p t e.val j.val) u
  gate := fun t => (u.mem "D" (gateAddress c p t)).read .real
  seed := raw "Seed" .real (fun e j => seedAddress c p s e.val j.val) u
  incoming := fun t => raw "DS" .real (fun e j => address c p t e.val j.val) u
  saved := fun t => raw "S" .fp32 (fun e j => address c p t e.val j.val) u

#specNonCircular forward avoiding [FP.Structural.exec,FP.Structural.run]
#specNonCircular reverse avoiding [FP.Structural.exec,FP.Structural.run]
#specNonCircular contraction avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReference
