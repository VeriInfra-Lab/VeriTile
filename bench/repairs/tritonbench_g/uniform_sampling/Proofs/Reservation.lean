import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Model

namespace VeriTile.Bench.Repairs.UniformSampling.ReservationProof
open VeriTile.Triton

theorem block_properties (n : Nat) :
    (block n=512 ∨ block n=1024) ∧ 0<block n ∧ block n∣2^32 ∧ 4∣block n := by
  unfold block VeriTile.Bench.TritonBenchG.UniformSampling.SourceDomain.wrapperBlock
  split <;> norm_num

theorem alignment_bounds (offset B : Nat) (hB:0<B) :
    offset≤alignedStart offset B ∧ alignedStart offset B<offset+B := by
  have h:=Nat.mod_lt (B-offset%B) hB
  unfold alignedStart
  omega

theorem alignment (offset B : Nat) (hB:0<B) : B∣alignedStart offset B := by
  have hr:=Nat.mod_lt offset hB
  by_cases hz:offset%B=0
  · simpa [alignedStart,hz] using (Nat.dvd_of_mod_eq_zero hz)
  · have hs : (B-offset%B)%B=B-offset%B := Nat.mod_eq_of_lt (by omega)
    have he : offset+(B-offset%B)=(offset/B+1)*B := by
      have h:=Nat.mod_add_div offset B
      have hs:=Nat.sub_add_cancel hr.le
      nlinarith
    simp only [alignedStart,hs,he]
    exact dvd_mul_left B (offset/B+1)

theorem align_add (offset B k : Nat) (hB:0<B) (h:B∣offset) :
    alignedStart (offset+k*B) B=offset+k*B := by
  have hm : (offset+k*B)%B=0 := by simp [Nat.mod_eq_zero_of_dvd h]
  simp [alignedStart,hm]

theorem success {owner B count : Nat} {g : Generator} {r : Reservation} {g' : Generator}
    (h:reserve owner B count g=.ok (r,g')) :
    g.device=owner ∧ (B=512 ∨ B=1024) ∧ 0<count ∧ B∣count ∧
    r.seed=g.seed ∧ r.start=alignedStart g.offset B ∧ r.stop=r.start+count ∧
    g'={g with offset:=r.stop} ∧ r.stop<2^64 := by
  unfold reserve at h
  split at h <;> simp_all
  split at h <;> simp_all
  split at h <;> simp_all
  split at h <;> simp_all
  obtain ⟨rfl,rfl⟩:=h
  simp_all [Nat.dvd_iff_mod_eq_zero]
  omega

theorem reserve_exists (owner B count : Nat) (g : Generator)
    (hd:g.device=owner) (hb:B=512 ∨ B=1024) (hc:0<count) (hm:B∣count)
    (fit:alignedStart g.offset B+count<2^64) :
    reserve owner B count g=.ok
      (⟨g.seed,alignedStart g.offset B,alignedStart g.offset B+count⟩,
        {g with offset:=alignedStart g.offset B+count}) := by
  simp [reserve,hd,hb,Nat.ne_of_gt hc,Nat.mod_eq_zero_of_dvd hm]
  omega

theorem evaluated_reserved {owner B count : Nat} {g : Generator} {r : Reservation} {g' : Generator}
    (h:reserve owner B count g=.ok (r,g')) {i : Nat} (hi:i<count) :
    g.offset≤r.start+i ∧ r.start+i<r.stop ∧ r.start+i<2^64 := by
  have hs:=success h
  have hb:0<B:=by rcases hs.2.1 with rfl|rfl <;> decide
  have ha:=alignment_bounds g.offset B hb
  rcases hs with ⟨_,_,_,_,_,hstart,hstop,_,hfit⟩
  omega

theorem consecutive_disjoint {owner B C count next : Nat} {g g' g'' : Generator}
    {r s : Reservation} (h:reserve owner B count g=.ok (r,g'))
    (k:reserve owner C next g'=.ok (s,g'')) {i j : Nat} (hi:i<count) (hj:j<next) :
    r.start+i<s.start+j := by
  have hr:=success h
  have hs:=success k
  have hc:0<C:=by rcases hs.2.1 with rfl|rfl <;> decide
  have ha:=alignment_bounds g'.offset C hc
  have hg:g'.offset=r.stop:=by rw [hr.2.2.2.2.2.2.2.1]
  have hv:=evaluated_reserved h hi
  rw [←hs.2.2.2.2.2.1] at ha
  omega

/-- The selected generator is the only entry modified by the mutex transaction. -/
def commit (generators : Nat→Generator) (selected : Nat) (next : Generator) : Nat→Generator :=
  Function.update generators selected next

theorem other_generator (generators : Nat→Generator) (selected other : Nat) (next : Generator)
    (h:other≠selected) : commit generators selected next other=generators other := by
  simp [commit,Function.update_of_ne h]

#axiomsClean alignment
#axiomsClean success
#axiomsClean evaluated_reserved
#axiomsClean consecutive_disjoint
#axiomsClean other_generator
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.UniformSampling.ReservationProof
