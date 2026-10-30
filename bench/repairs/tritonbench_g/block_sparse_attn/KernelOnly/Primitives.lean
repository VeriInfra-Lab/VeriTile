import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Source
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

/-- Task-local instruction and layout vocabulary, independent of legacy
execution-model object files. This selected kernel uses only MMA m16n8k8. -/
inductive Instruction where
  | mma16n8k8
  deriving DecidableEq
structure Ops (α : Type) where
  zero : α
  negInf : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  max : α → α → α
  exp : α → α
  div : α → α → α
  mma8 : Instruction → (Fin 8 → α) → (Fin 8 → α) → α → α

inductive Tree (n : Nat) where
  | leaf (i : Fin n)
  | merge (left right : Tree n)
def Tree.leaves : Tree n → List (Fin n)
  | .leaf i => [i]
  | .merge l r => l.leaves++r.leaves
def Tree.eval (op : α → α → α) (f : Fin n → α) : Tree n → α
  | .leaf i => f i
  | .merge l r => op (l.eval op f) (r.eval op f)
def owner (column : Nat) : Fin 4 :=
  ⟨column%8/2,by have := Nat.mod_lt column (by decide : 0<8); omega⟩

namespace SM90B16
def localTree (o : Fin 4) : Tree 16 :=
  .merge (.merge (.merge (.leaf ⟨2*o.val,by omega⟩)
    (.leaf ⟨2*o.val+1,by omega⟩)) (.leaf ⟨8+2*o.val,by omega⟩))
      (.leaf ⟨9+2*o.val,by omega⟩)
def xor2 (o : Fin 4) : Fin 4 := Fin.ofNat 4 (Nat.xor o.val 2)
def xor1 (o : Fin 4) : Fin 4 := Fin.ofNat 4 (Nat.xor o.val 1)
def stage2 (o : Fin 4) : Tree 16 := .merge (localTree o) (localTree (xor2 o))
def tree (o : Fin 4) : Tree 16 := .merge (stage2 o) (stage2 (xor1 o))
theorem coverage (o : Fin 4) : (tree o).leaves.Perm (List.finRange 16) := by
  fin_cases o <;> decide
#axiomsClean coverage
end SM90B16
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
