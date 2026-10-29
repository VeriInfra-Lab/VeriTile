import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Reduction
set_option maxRecDepth 20000
set_option maxHeartbeats 4000000
inductive Tree where
  | leaf (index : Fin 128)
  | merge (a b : Tree)
  | shuffle (mask : Nat) (a : Tree)

def Tree.leaves : Tree → List Nat
  | .leaf i => [i.val]
  | .merge a b => a.leaves ++ b.leaves
  | .shuffle _ a => a.leaves

def Tree.eval (op : α → α → α) (input : Nat → α) : Tree → α
  | .leaf i => input i.val
  | .merge a b => op (a.eval op input) (b.eval op input)
  | .shuffle _ a => a.eval op input

def localIndex (owner : Fin 4) (j : Fin 32) : Fin 128 :=
  ⟨j.val/2*8+2*owner.val+j.val%2,by have:=owner.isLt;have:=j.isLt;omega⟩
def localTree (owner : Fin 4) : Tree :=
  (List.finRange 31).foldl (fun t j => .merge t (.leaf (localIndex owner ⟨j.val+1,by omega⟩)))
    (.leaf (localIndex owner ⟨0,by decide⟩))
def xorOwner (owner : Fin 4) (mask : Nat) : Fin 4 :=
  ⟨(owner.val ^^^ mask)%4,Nat.mod_lt _ (by decide)⟩
def plan (owner : Fin 4) : Tree :=
  .merge (.merge (localTree owner) (.shuffle 2 (localTree (xorOwner owner 2))))
    (.shuffle 1 (.merge (localTree (xorOwner owner 1))
      (.shuffle 2 (localTree (xorOwner owner 3)))))
def owner (column : Nat) : Fin 4 := ⟨column%8/2,by have:=Nat.mod_lt column (by decide : 0<8);omega⟩
theorem coverage (o : Fin 4) : (plan o).leaves.Perm (List.range 128) := by
  fin_cases o <;> decide

theorem tree_sum (t : Tree) (f : Nat → ℝ) :
    t.eval (·+·) f = (t.leaves.map f).sum := by
  induction t with
  | leaf i => simp [Tree.eval,Tree.leaves]
  | merge a b ha hb => simp [Tree.eval,Tree.leaves,ha,hb,List.sum_append]
  | shuffle mask a ha => exact ha

theorem sum_exact (o : Fin 4) (f : Nat → ℝ) :
    (plan o).eval (·+·) f = ((List.range 128).map f).sum := by
  rw [tree_sum]
  exact ((coverage o).map f).sum_eq
#axiomsClean coverage
#axiomsClean sum_exact
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Reduction
