import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelGrid
open KernelProgram KernelIndex
noncomputable section

def program (c : Shape) (h : Fin c.heads) (k : Fin c.K) (v : Fin c.V) : Program c where
  head := h.val
  key := k.val/64
  value := v.val/16
  headInside := h.isLt
  keyInside := (coordinate_coverage c k.val v.val k.isLt v.isLt).1
  valueInside := (coordinate_coverage c k.val v.val k.isLt v.isLt).2.1
def keyLane {c : Shape} (k : Fin c.K) : Fin 64 := ⟨k.val%64,Nat.mod_lt _ (by decide)⟩
def valueLane {c : Shape} (v : Fin c.V) : Fin 16 := ⟨v.val%16,Nat.mod_lt _ (by decide)⟩

theorem coverage (c : Shape) (h : Fin c.heads) (t : Fin c.N) (k : Fin c.K) (v : Fin c.V) :
    address c (program c h k v) t.val (keyLane k).val (valueLane v).val=
      matrixOffset c h.val t.val k.val v.val := by
  rw [matrix_address c (program c h k v) t.val (keyLane k) (valueLane v)]
  simp only [program,keyLane,valueLane]
  congr 2 <;> omega

theorem unique (c : Shape) (p q : Program c) (t u : Fin c.N)
    (e a : Fin 64) (j b : Fin 16)
    (equal : address c p t.val e.val j.val=address c q u.val a.val b.val) :
    p.head=q.head ∧ t=u ∧ p.key=q.key ∧ e=a ∧ p.value=q.value ∧ j=b := by
  rw [matrix_address,matrix_address] at equal
  obtain ⟨hh,ht,hk,hv⟩ := matrix_injective c _ _ _ _ _ _ _ _ t.isLt
    (key_inside c p e) (value_inside c p j) u.isLt (key_inside c q a) (value_inside c q b) equal
  obtain ⟨hk,he⟩ := digit_injective 64 _ _ _ _ e.isLt a.isLt hk
  obtain ⟨hv,hj⟩ := digit_injective 16 _ _ _ _ j.isLt b.isLt hv
  exact ⟨hh,Fin.ext ht,hk,Fin.ext he,hv,Fin.ext hj⟩

#axiomsClean coverage
#axiomsClean unique
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelGrid
