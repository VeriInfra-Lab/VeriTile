import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Raw

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.RawStores
open VeriTile.Triton Geometry Host Ownership Raw
noncomputable section

/-- Exact whole-grid ordinary stores from the complete all-word graph.
Untouched regions, holes, inactive L, and tail allocation rows retain bits. -/
def stores (c : Call) (ops : Ops) (schedule : Schedule c) (memory : Memory) (scale : Word) : Memory := by
  classical
  exact fun rg off=>if h : ∃ idx i,active c i ∧ address c idx i=(rg,off) then
    output c ops schedule memory scale (Classical.choose h) (Classical.choose (Classical.choose_spec h))
  else memory rg off

theorem readback (c : Call) (hc : Valid c) (ops : Ops) (schedule : Schedule c)
    (memory : Memory) (scale : Word) (idx : GridIndex c.grid) (i : Lane c.BM c.D) (hi : active c i) :
    stores c ops schedule memory scale (address c idx i).1 (address c idx i).2=
      output c ops schedule memory scale idx i := by
  have hex : ∃ jdx j,active c j ∧ address c jdx j=((address c idx i).1,(address c idx i).2) := ⟨idx,i,hi,rfl⟩
  rw [stores,dif_pos hex]
  have hj:=(Classical.choose_spec (Classical.choose_spec hex))
  have hu:=Ownership.unique c hc (Classical.choose hex) idx
    (Classical.choose (Classical.choose_spec hex)) i hj.2
  exact congrArg₂ (output c ops schedule memory scale) hu.1 hu.2

theorem frame (c : Call) (ops : Ops) (schedule : Schedule c) (memory : Memory) (scale : Word)
    (rg : RegionName) (off : Nat) (hn : ¬∃ idx i,active c i ∧ address c idx i=(rg,off)) :
    stores c ops schedule memory scale rg off=memory rg off := by rw [stores,dif_neg hn]

theorem end_L (c : Call) (hf : c.finish≠0) (ops : Ops) (schedule : Schedule c)
    (memory : Memory) (scale : Word) (off : Nat) : stores c ops schedule memory scale "L" off=memory "L" off := by
  apply frame
  rintro ⟨idx,i,hi,he⟩
  cases i <;> simp [address,active,hf] at hi he

abbrev Byte:=BitVec 8

def byte (word : Word) (lane : Fin 4) : Byte:=BitVec.extractLsb' (8*lane.val) 8 word

def bytes (c : Call) (ops : Ops) (schedule : Schedule c) (memory : Memory) (scale : Word)
    (rg : RegionName) (off : Nat) (lane : Fin 4) : Byte:=byte (stores c ops schedule memory scale rg off) lane

theorem whole_bytes (c : Call) (a b : Ops) (h : LocalLaws a b) (schedule : Schedule c)
    (memory : Memory) (scale : Word) : bytes c a schedule memory scale=bytes c b schedule memory scale := by
  rw [operations_eq a b h]

#print axioms readback
#print axioms whole_bytes
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.RawStores
