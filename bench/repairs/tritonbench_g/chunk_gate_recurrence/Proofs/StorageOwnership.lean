import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Guards

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
open Machine

theorem coordinate_injective (c : Shape) (p j p' j' : Nat)
    (hj : j < 1024) (hj' : j' < 1024)
    (head : p/c.tiles=p'/c.tiles)
    (key : (p%c.tiles/c.VT)*64+j/16=(p'%c.tiles/c.VT)*64+j'/16)
    (value : (p%c.tiles%c.VT)*16+j%16=(p'%c.tiles%c.VT)*16+j'%16) : p=p' ∧ j=j' := by
  have k := digit_injective 64 _ _ _ _ (by omega : j/16 < 64) (by omega : j'/16 < 64) key
  have v := digit_injective 16 _ _ _ _ (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide)) value
  have r1 := Nat.div_add_mod (p%c.tiles) c.VT
  have r2 := Nat.div_add_mod (p'%c.tiles) c.VT
  rw [k.1,v.1] at r1
  have rem : p%c.tiles=p'%c.tiles := by omega
  have h1 := Nat.div_add_mod p c.tiles
  have h2 := Nat.div_add_mod p' c.tiles
  rw [head,rem] at h1
  constructor
  · omega
  · have a := Nat.div_add_mod j 16
    have b := Nat.div_add_mod j' 16
    rw [k.2,v.2] at a
    omega

theorem active_written (c : Shape) (seed : Bool) (r : Region) (p t j : Nat)
    (write : Source.writes c seed r p t j) : Written r ∧ Active c seed r p t j := by
  rcases write with h|h|h|h|h
  · obtain ⟨rfl,hp,ht,hj⟩ := h
    exact ⟨Or.inl rfl,⟨hp,ht,hj⟩⟩
  · obtain ⟨rfl,hp,ht,hj⟩ := h
    exact ⟨Or.inr (Or.inl rfl),⟨hp,ht,hj⟩⟩
  · obtain ⟨rfl,hp,ht,rfl⟩ := h
    exact ⟨Or.inr (Or.inr (Or.inl rfl)),⟨hp,ht,rfl⟩⟩
  · obtain ⟨rfl,hp,rfl,hj,hs⟩ := h
    exact ⟨Or.inr (Or.inr (Or.inr (Or.inl rfl))),⟨hs,hp,rfl,hj⟩⟩
  · obtain ⟨rfl,hp,ht,rfl⟩ := h
    exact ⟨Or.inr (Or.inr (Or.inr (Or.inr rfl))),⟨hp,ht,rfl⟩⟩

theorem offset_injective (c : Shape) (seed : Bool) (abi : ABI) (r : Region) (p t j p' t' j' : Nat)
    (written : Written r) (active : Active c seed r p t j) (active' : Active c seed r p' t' j')
    (same : offset c abi r p t j=offset c abi r p' t' j') : p=p' ∧ t=t' ∧ j=j' := by
  rcases written with rfl|rfl|rfl|rfl|rfl
  all_goals simp only [offset] at same
  · obtain ⟨hp,ht,hj⟩ := active
    obtain ⟨hp',ht',hj'⟩ := active'
    have coord := coordinates c p j hp hj
    have coord' := coordinates c p' j' hp' hj'
    have eq : matrixOffset c (p/c.tiles) t ((p%c.tiles/c.VT)*64+j/16) ((p%c.tiles%c.VT)*16+j%16)=
        matrixOffset c (p'/c.tiles) t' ((p'%c.tiles/c.VT)*64+j'/16) ((p'%c.tiles%c.VT)*16+j'%16) := by exact_mod_cast same
    have fields := matrix_injective c _ _ _ _ _ _ _ _ ht coord.2.2.2.1 coord.2.2.2.2 ht' coord'.2.2.2.1 coord'.2.2.2.2 eq
    have pairs := coordinate_injective c p j p' j' hj hj' fields.1 fields.2.2.1 fields.2.2.2
    exact ⟨pairs.1,fields.2.1,pairs.2⟩
  · obtain ⟨hp,ht,hj⟩ := active
    obtain ⟨hp',ht',hj'⟩ := active'
    have coord := coordinates c p j hp hj
    have coord' := coordinates c p' j' hp' hj'
    have eq : matrixOffset c (p/c.tiles) t ((p%c.tiles/c.VT)*64+j/16) ((p%c.tiles%c.VT)*16+j%16)=
        matrixOffset c (p'/c.tiles) t' ((p'%c.tiles/c.VT)*64+j'/16) ((p'%c.tiles%c.VT)*16+j'%16) := by exact_mod_cast same
    have fields := matrix_injective c _ _ _ _ _ _ _ _ ht coord.2.2.2.1 coord.2.2.2.2 ht' coord'.2.2.2.1 coord'.2.2.2.2 eq
    have pairs := coordinate_injective c p j p' j' hj hj' fields.1 fields.2.2.1 fields.2.2.2
    exact ⟨pairs.1,fields.2.1,pairs.2⟩
  · obtain ⟨hp,ht,rfl⟩ := active
    obtain ⟨hp',ht',rfl⟩ := active'
    have rem := Nat.mod_lt p c.tiles_pos
    have rem' := Nat.mod_lt p' c.tiles_pos
    have decode : (p%c.tiles/c.VT)*c.VT+p%c.tiles%c.VT=p%c.tiles := by
      simpa only [Nat.mul_comm] using Nat.div_add_mod (p%c.tiles) c.VT
    have decode' : (p'%c.tiles/c.VT)*c.VT+p'%c.tiles%c.VT=p'%c.tiles := by
      simpa only [Nat.mul_comm] using Nat.div_add_mod (p'%c.tiles) c.VT
    have eq : (p/c.tiles*c.N+t)*c.tiles+p%c.tiles=(p'/c.tiles*c.N+t')*c.tiles+p'%c.tiles := by
      norm_cast at same
      simpa only [scratchOffset,Nat.add_assoc,decode,decode'] using same
    have tile := digit_injective c.tiles _ _ _ _ rem rem' eq
    have head := digit_injective c.N _ _ _ _ ht ht' tile.1
    have a := Nat.div_add_mod p c.tiles
    have b := Nat.div_add_mod p' c.tiles
    rw [head.1,tile.2] at a
    exact ⟨by omega,head.2,rfl⟩
  · obtain ⟨_,hp,rfl,hj⟩ := active
    obtain ⟨_,hp',rfl,hj'⟩ := active'
    have coord := coordinates c p j hp hj
    have coord' := coordinates c p' j' hp' hj'
    have eq : seedDenseOffset c (p/c.tiles) ((p%c.tiles/c.VT)*64+j/16) ((p%c.tiles%c.VT)*16+j%16)=
        seedDenseOffset c (p'/c.tiles) ((p'%c.tiles/c.VT)*64+j'/16) ((p'%c.tiles%c.VT)*16+j'%16) := by exact_mod_cast same
    have value := digit_injective c.V _ _ _ _ coord.2.2.2.2 coord'.2.2.2.2 eq
    have key := digit_injective c.K _ _ _ _ coord.2.2.2.1 coord'.2.2.2.1 value.1
    have pairs := coordinate_injective c p j p' j' hj hj' key.1 key.2 value.2
    exact ⟨pairs.1,rfl,pairs.2⟩
  · obtain ⟨_,ht,rfl⟩ := active
    obtain ⟨_,ht',rfl⟩ := active'
    have eq : p*c.N+t=p'*c.N+t' := by exact_mod_cast same
    have pair := digit_injective c.N _ _ _ _ ht ht' eq
    exact ⟨pair.1,pair.2,rfl⟩

#axiomsClean coordinate_injective
#axiomsClean active_written
#axiomsClean offset_injective
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
