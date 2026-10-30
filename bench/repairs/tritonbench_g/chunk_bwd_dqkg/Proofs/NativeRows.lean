import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Native

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
noncomputable section
set_option maxHeartbeats 2000000

inductive Width where | w16 | w32 | w64
  deriving DecidableEq
def Width.size : Width → Nat | .w16=>16 | .w32=>32 | .w64=>64

inductive Tree (n : Nat) where
  | leaf (index : Fin n)
  | merge (a b : Tree n)
  | shuffle (mask : Nat) (a : Tree n)

def Tree.leaves : Tree n → List Nat
  | .leaf i=>[i.val]
  | .merge a b=>a.leaves++b.leaves
  | .shuffle _ a=>a.leaves
def Tree.value (input : Fin n → Graph) : Tree n → Graph
  | .leaf i=>input i
  | .merge a b=>.add32 (a.value input) (b.value input)
  | .shuffle mask a=>.shuffle mask (a.value input)

def rowLocalIndex (w : Width) (owner : Fin 4) (j : Fin (w.size/4)) : Fin w.size :=
  ⟨j.val/2*8+2*owner.val+j.val%2,by
    have ho:=owner.isLt
    have hj:=j.isLt
    cases w <;> dsimp [Width.size] at * <;> omega⟩

def rowLocalTree (w : Width) (owner : Fin 4) : Tree w.size :=
  (List.finRange (w.size/4-2)).foldl
    (fun tree j=>let next:=Tree.leaf (rowLocalIndex w owner ⟨j.val+2,by
      have hj:=j.isLt
      cases w <;> dsimp [Width.size] at * <;> omega⟩)
      if j.val=0 then .merge tree next else .merge next tree)
    (.merge (.leaf (rowLocalIndex w owner ⟨0,by cases w <;> decide⟩))
      (.leaf (rowLocalIndex w owner ⟨1,by cases w <;> decide⟩)))

def xorOwner (owner : Fin 4) (mask : Nat) : Fin 4 :=
  ⟨(owner.val ^^^ mask)%4,Nat.mod_lt _ (by decide)⟩

/-- Ordered row reduction: add the first pair, append the third term on the
right, then prepend each remaining term on the left; XOR2 then XOR1. The PTX
checker binds terms to the final MMA destination group and coordinate. This
local order is not a left fold, even though the Real leaf coverage is equal. -/
def rowPlan (w : Width) (owner : Fin 4) : Tree w.size :=
  .merge (.merge (rowLocalTree w owner) (.shuffle 2 (rowLocalTree w (xorOwner owner 2))))
    (.shuffle 1 (.merge (rowLocalTree w (xorOwner owner 1))
      (.shuffle 2 (rowLocalTree w (xorOwner owner 3)))))

theorem row_coverage (w : Width) (owner : Fin 4) :
    (rowPlan w owner).leaves.Perm (List.range w.size) := by
  cases w <;> fin_cases owner <;> decide

def rowSum (w : Width) (owner : Fin 4) (input : Fin w.size → Graph) : Graph :=
  (rowPlan w owner).value input

#axiomsClean row_coverage
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
