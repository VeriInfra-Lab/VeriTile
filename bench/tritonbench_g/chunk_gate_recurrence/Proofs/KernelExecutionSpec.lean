import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReference
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram
noncomputable section
variable {α : Type} [Inhabited α]

def inputRoles (kind : Kind) (seed : Bool) : List RegionName := match kind with
  | .forward => ["S","D"]++if seed then ["Seed"] else []
  | .backward => ["S","D","DS"]
def outputRoles : Kind → List RegionName
  | .forward => ["O"]
  | .backward => ["DI","DG","DL"]
def argumentRoles (kind : Kind) : List RegionName := match kind with
  | .forward => ["S","D","O","Seed"]
  | .backward => ["S","D","DI","DG","DL","DS"]
def dtype (kind : Kind) (r : RegionName) : FloatDType :=
  if kind=.backward ∧ (r="S" ∨ r="DL") then .fp32 else .real

def expected (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (r : RegionName) (t : Nat) (i : Lane) : α :=
  let a := KernelReference.read c s p u
  match kind with
  | .forward => KernelReference.forward M a seed t i
  | .backward =>
    if r="DI" then KernelReference.reverse M a c.N (c.N-1-t) i
    else if r="DG" then KernelReference.contraction M (KernelReference.reverse M a c.N (c.N-1-t)) (a.saved t)
    else KernelReference.reverse M a c.N c.N i

def Output (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u v : State α) : Prop := match kind with
  | .forward => ∀ (t : Fin c.N) (e : Fin 64) (j : Fin 16),
      v.mem "O" (address c p t.val e.val j.val)=
        KernelReference.stored M .real (expected M .forward c s seed p u "O" t.val (e,j,PUnit.unit))
  | .backward =>
    (∀ (t : Fin (c.N-1)) (e : Fin 64) (j : Fin 16),
      v.mem "DI" (address c p t.val e.val j.val)=
        KernelReference.stored M .real (expected M .backward c s seed p u "DI" t.val (e,j,PUnit.unit))) ∧
    (∀ t : Fin (c.N-1),v.mem "DG" (gradientAddress c p t.val)=
      KernelReference.stored M .real (expected M .backward c s seed p u "DG" t.val (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit))) ∧
    (∀ (e : Fin 64) (j : Fin 16),v.mem "DL" (dseedAddress c p e.val j.val)=
      KernelReference.stored M .fp32 (expected M .backward c s seed p u "DL" 0 (e,j,PUnit.unit)))

structure Facts (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u v : State α) : Prop where
  success : FP.Structural.exec (engine M) (source kind c s seed) u=some v
  outputs : Output M kind c s seed p u v
  frame : KernelMemory.Frame kind c p u v

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelExecution
