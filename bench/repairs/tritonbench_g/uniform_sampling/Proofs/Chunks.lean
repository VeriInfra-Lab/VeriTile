import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Reservation

namespace VeriTile.Bench.Repairs.UniformSampling.ChunkProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.UniformSampling

private theorem progress (start done remaining : Nat) (hr:0<remaining) :
    0<min remaining (2^32-(start+done)%2^32) := by
  have h:=Nat.mod_lt (start+done) (by decide : 0<2^32)
  omega

theorem member (start remaining done : Nat) {c : Chunk} (hc:c∈chunksAux start remaining done) :
    done≤c.done ∧ c.done+c.count≤done+remaining ∧ 0<c.count ∧
    c.count≤2^32-(start+c.done)%2^32 := by
  induction remaining using Nat.strong_induction_on generalizing done with
  | h remaining ih =>
    cases remaining with
    |zero=>simp [chunksAux] at hc
    |succ r=>
      rw [chunksAux] at hc
      let count:=min (r+1) (2^32-(start+done)%2^32)
      have hp:0<count:=progress start done (r+1) (by omega)
      have hl:count≤r+1:=Nat.min_le_left _ _
      rcases List.mem_cons.mp hc with he|ht
      · subst c
        dsimp
        exact ⟨le_rfl,by omega,hp,Nat.min_le_right _ _⟩
      · have hs:=ih (r+1-count) (by omega) (done+count) ht
        exact ⟨by omega,by omega,hs.2.2.1,hs.2.2.2⟩

theorem cover (start remaining done k : Nat) (lo:done≤k) (hi:k<done+remaining) :
    ∃ c,c∈chunksAux start remaining done ∧ c.done≤k ∧ k<c.done+c.count := by
  induction remaining using Nat.strong_induction_on generalizing done with
  | h remaining ih =>
    cases remaining with
    |zero=>omega
    |succ r=>
      let count:=min (r+1) (2^32-(start+done)%2^32)
      have hp:0<count:=progress start done (r+1) (by omega)
      have hl:count≤r+1:=Nat.min_le_left _ _
      by_cases hk:k<done+count
      · exact ⟨⟨done,count⟩,by rw [chunksAux];exact List.mem_cons_self,lo,hk⟩
      · obtain ⟨c,hc,hlo,hhi⟩:=ih (r+1-count) (by omega) (done+count) (by omega) (by omega)
        exact ⟨c,by rw [chunksAux];exact List.mem_cons_of_mem _ hc,hlo,hhi⟩

theorem unique (start remaining done k : Nat) {c d : Chunk}
    (hc:c∈chunksAux start remaining done) (hd:d∈chunksAux start remaining done)
    (cl:c.done≤k) (ch:k<c.done+c.count) (dl:d.done≤k) (dh:k<d.done+d.count) : c=d := by
  induction remaining using Nat.strong_induction_on generalizing done with
  | h remaining ih =>
    cases remaining with
    |zero=>simp [chunksAux] at hc
    |succ r=>
      rw [chunksAux] at hc hd
      let count:=min (r+1) (2^32-(start+done)%2^32)
      have hp:0<count:=progress start done (r+1) (by omega)
      rcases List.mem_cons.mp hc with hc|hc <;> rcases List.mem_cons.mp hd with hd|hd
      · exact hc.trans hd.symm
      · subst c
        have hm:=member start (r+1-count) (done+count) hd
        dsimp at ch
        omega
      · subst d
        have hm:=member start (r+1-count) (done+count) hc
        dsimp at dh
        omega
      · exact ih (r+1-count) (by omega) (done+count) hc hd

private theorem count_divisible (B start done remaining : Nat)
    (hb:B∣2^32) (hs:B∣start) (hd:B∣done) (hr:B∣remaining) :
    B∣min remaining (2^32-(start+done)%2^32) := by
  have hm:B∣(start+done)%2^32:=(Nat.dvd_mod_iff hb).mpr (Nat.dvd_add hs hd)
  have ht:B∣2^32-(start+done)%2^32:=Nat.dvd_sub hb hm
  exact (min_choice remaining (2^32-(start+done)%2^32)).elim (fun h=>h.symm ▸ hr) (fun h=>h.symm ▸ ht)

theorem divisible (B start remaining done : Nat) (hb:B∣2^32)
    (hs:B∣start) (hr:B∣remaining) (hd:B∣done) {c : Chunk}
    (hc:c∈chunksAux start remaining done) : B∣c.done ∧ B∣c.count := by
  induction remaining using Nat.strong_induction_on generalizing done with
  | h remaining ih =>
    cases remaining with
    |zero=>simp [chunksAux] at hc
    |succ r=>
      rw [chunksAux] at hc
      let count:=min (r+1) (2^32-(start+done)%2^32)
      have hp:0<count:=progress start done (r+1) (by omega)
      have hcdiv:B∣count:=count_divisible B start done (r+1) hb hs hd hr
      rcases List.mem_cons.mp hc with he|ht
      · subst c;exact ⟨hd,hcdiv⟩
      · exact ih (r+1-count) (by omega) (done+count) (Nat.dvd_sub hr hcdiv)
          (Nat.dvd_add hd hcdiv) ht

theorem total_covers (n : Nat) : n≤4*counterCount n := by
  have hB:0<block n:=(ReservationProof.block_properties n).2.1
  have h:n≤(4*block n)*grid n (block n):=
    (ceilDiv_le_iff_le_mul (by omega : 0<4*block n)).mp le_rfl
  unfold counterCount
  nlinarith

theorem total_positive {n : Nat} (hn:0<n) : 0<counterCount n := by
  have h:=total_covers n
  omega

theorem chunks_properties (n start : Nat) (hs:block n∣start) {c : Chunk} (hc:c∈chunks n start) :
    c.done+c.count≤counterCount n ∧ 0<c.count ∧
    (start+c.done)%2^32+c.count≤2^32 ∧ block n∣c.done ∧ block n∣c.count := by
  have hm:=member start (counterCount n) 0 hc
  have hb:=ReservationProof.block_properties n
  have hd:=divisible (block n) start (counterCount n) 0 hb.2.2.1 hs
    (dvd_mul_left _ _) (dvd_zero _) hc
  have hr:=Nat.mod_lt (start+c.done) (by decide : 0<2^32)
  exact ⟨by omega,hm.2.2.1,by omega,hd.1,hd.2⟩

theorem empty : chunks 0 start=[] := by
  simp [chunks,counterCount,SourceDomain.grid_zero,chunksAux]

#axiomsClean member
#axiomsClean cover
#axiomsClean unique
#axiomsClean divisible
#axiomsClean chunks_properties
#axiomsClean total_covers
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.UniformSampling.ChunkProof
