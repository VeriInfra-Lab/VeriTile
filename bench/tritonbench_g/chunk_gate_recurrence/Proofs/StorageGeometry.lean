import bench.tritonbench_g.chunk_gate_recurrence.Proofs.Safety

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
open Machine

def extent (c : Shape) (s : SeedStrides) (r : Region) : Nat := match r with
  | .x | .output | .incoming | .dx => c.heads*c.N*c.K*c.V
  | .gate | .dgate => c.heads*c.N
  | .seed => (c.B-1)*s.batch+(c.H-1)*s.head+(c.K-1)*s.key+(c.V-1)*s.value+1
  | .dseed => c.heads*c.K*c.V
  | .tileGate => c.heads*c.N*c.tiles
  | .caller => 0

theorem coordinates (c : Shape) (p j : Nat) (hp : p < c.programs) (hj : j < 1024) :
    p/c.tiles < c.heads ∧ p%c.tiles/c.VT < c.KT ∧ p%c.tiles%c.VT < c.VT ∧
    (p%c.tiles/c.VT)*64+j/16 < c.K ∧ (p%c.tiles%c.VT)*16+j%16 < c.V := by
  have rem := Nat.mod_lt p c.tiles_pos
  have tile : p%c.tiles/c.VT < c.KT :=
    (Nat.div_lt_iff_lt_mul c.vpos).mpr (by simpa [Shape.tiles] using rem)
  have lane : j/16 < 64 := by omega
  have value := Nat.mod_lt (p%c.tiles) c.vpos
  exact ⟨Safety.head_bound c p hp,tile,value,
    Safety.mixed_bound _ _ _ _ tile lane,
    Safety.mixed_bound _ _ _ _ value (Nat.mod_lt _ (by decide))⟩

theorem seed_bound (c : Shape) (s : SeedStrides) (h k v : Nat)
    (hh : h < c.heads) (hk : k < c.K) (hv : v < c.V) : seedOffset c s h k v < extent c s .seed := by
  have batch : h/c.H < c.B := (Nat.div_lt_iff_lt_mul c.hpos).mpr hh
  have head := Nat.mod_lt h c.hpos
  have bmul := Nat.mul_le_mul_right s.batch (show h/c.H ≤ c.B-1 by omega)
  have hmul := Nat.mul_le_mul_right s.head (show h%c.H ≤ c.H-1 by omega)
  have kmul := Nat.mul_le_mul_right s.key (show k ≤ c.K-1 by omega)
  have vmul := Nat.mul_le_mul_right s.value (show v ≤ c.V-1 by omega)
  simp only [seedOffset,extent]
  omega

theorem seed_dense_bound (c : Shape) (h k v : Nat) (hh : h < c.heads) (hk : k < c.K) (hv : v < c.V) :
    seedDenseOffset c h k v < c.heads*c.K*c.V :=
  Safety.mixed_bound _ _ _ _ (Safety.mixed_bound _ _ _ _ hh hk) hv

theorem scratch_bound (c : Shape) (h t tk tv : Nat)
    (hh : h < c.heads) (ht : t < c.N) (hk : tk < c.KT) (hv : tv < c.VT) :
    scratchOffset c h t tk tv < c.heads*c.N*c.tiles := by
  simpa only [scratchOffset,Shape.tiles,Nat.add_assoc] using
    Safety.mixed_bound _ _ _ _ (Safety.mixed_bound _ _ _ _ hh ht) (Safety.mixed_bound _ _ _ _ hk hv)

theorem active_offset (c : Shape) (seed : Bool) (abi : ABI) (r : Region) (p t j : Nat)
    (active : Active c seed r p t j) :
    0 ≤ offset c abi r p t j ∧ offset c abi r p t j < (extent c abi.seedStrides r : Int) := by
  cases r with
  | x | output | incoming | dx =>
    obtain ⟨hp,ht,hj⟩ := active
    have coord := coordinates c p j hp hj
    have bound := main_bound c _ t _ _ coord.1 ht coord.2.2.2.1 coord.2.2.2.2
    simpa only [offset,extent] using (show 0 ≤ (matrixOffset c (p/c.tiles) t
      ((p%c.tiles/c.VT)*64+j/16) ((p%c.tiles%c.VT)*16+j%16) : Int) ∧
      (matrixOffset c (p/c.tiles) t ((p%c.tiles/c.VT)*64+j/16) ((p%c.tiles%c.VT)*16+j%16) : Int) <
      (c.heads*c.N*c.K*c.V : Nat) from ⟨Int.natCast_nonneg _,by exact_mod_cast bound⟩)
  | seed =>
    obtain ⟨_,_,hp,_,hj⟩ := active
    have coord := coordinates c p j hp hj
    have bound := seed_bound c abi.seedStrides _ _ _ coord.1 coord.2.2.2.1 coord.2.2.2.2
    simp only [offset,extent]
    exact ⟨Int.natCast_nonneg _,by exact_mod_cast bound⟩
  | dseed =>
    obtain ⟨_,hp,_,hj⟩ := active
    have coord := coordinates c p j hp hj
    have bound := seed_dense_bound c _ _ _ coord.1 coord.2.2.2.1 coord.2.2.2.2
    simp only [offset,extent]
    exact ⟨Int.natCast_nonneg _,by exact_mod_cast bound⟩
  | tileGate =>
    obtain ⟨hp,ht,_⟩ := active
    have coord := coordinates c p 0 hp (by decide)
    have bound := scratch_bound c _ t _ _ coord.1 ht coord.2.1 coord.2.2.1
    simp only [offset,extent]
    exact ⟨Int.natCast_nonneg _,by exact_mod_cast bound⟩
  | gate | dgate =>
    obtain ⟨hp,ht,_⟩ := active
    have bound := Safety.mixed_bound _ _ _ _ hp ht
    simp only [offset,extent]
    exact ⟨Int.natCast_nonneg _,by exact_mod_cast bound⟩
  | caller => cases active

#axiomsClean coordinates
#axiomsClean active_offset
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
