import bench.tritonbench_g.streamk_matmul.Proofs.Reference
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Protocol
open Plan
set_option maxHeartbeats 12000000

theorem member_start (I stop start : Nat) (hI : 0 < I) (c : Chunk)
    (hc : c ∈ chunks I stop start hI) : c.lo=start ∨ c.lo%I=0 := by
  by_cases h:start < stop
  · rw [chunks_cons I stop start hI h] at hc
    rcases List.mem_cons.mp hc with rfl|hc
    · exact Or.inl rfl
    · have ih:=member_start I stop (next I stop start) hI c hc
      rcases ih with he|he
      · right
        have hs:=member_shape I stop (next I stop start) hI c hc
        have hn:next I stop start < stop:=by omega
        have hb:next I stop start=boundary I start:=by
          rw [next_eq I stop start hI] at hn ⊢
          omega
        rw [he,hb,boundary,Nat.mul_mod_left]
      · exact Or.inr he
  · rw [chunks_nil I stop start hI (by omega)] at hc
    simp at hc
termination_by stop-start
decreasing_by
  have hp:=next_progress I stop start hI (by assumption)
  omega

/-- A source-emitted chunk is exactly the intersection of its program
interval with its tile, not an arbitrary partition satisfying coverage. -/
theorem intersection (I stop start : Nat) (hI : 0 < I) (c : Chunk)
    (hc : c ∈ chunks I stop start hI) :
    c.lo=max start (c.tile I*I) ∧ c.hi=min stop ((c.tile I+1)*I) := by
  have hs:=member_shape I stop start hI c hc
  constructor
  · rcases member_start I stop start hI c hc with he|he
    · have hd:=Nat.div_mul_le_self c.lo I
      change c.lo=max start (c.lo/I*I)
      omega
    · have hd:=Nat.div_add_mod c.lo I
      rw [he,Nat.add_zero,Nat.mul_comm] at hd
      change c.lo=max start (c.lo/I*I)
      omega
  · rw [hs.2.2.2,next_eq I stop c.lo hI,min_comm]
    rfl

/-- At most one chunk from a given program belongs to any specified tile. -/
theorem tile_chunk_unique (I stop start : Nat) (hI : 0 < I) (c d : Chunk)
    (hc : c ∈ chunks I stop start hI) (hd : d ∈ chunks I stop start hI)
    (ht : c.tile I=d.tile I) : c=d := by
  have hs:=intersection I stop start hI c hc
  have ht':=intersection I stop start hI d hd
  have hl:c.lo=d.lo:=by rw [hs.1,ht'.1,ht]
  have hh:c.hi=d.hi:=by rw [hs.2,ht'.2,ht]
  cases c with
  | mk clo chi =>
    cases d with
    | mk dlo dhi =>
      dsimp only at hl hh
      subst dlo
      subst dhi
      rfl

/-- Conversely every nonempty integer intersection is emitted by the actual
while loop. This closes the mathematical per-program decomposition. -/
theorem intersection_emitted (I stop start tile : Nat) (hI : 0 < I)
    (h : max start (tile*I) < min stop ((tile+1)*I)) :
    ∃ c ∈ chunks I stop start hI, c.tile I=tile := by
  have hlo:start ≤ max start (tile*I):=le_max_left _ _
  have hstop:max start (tile*I) < stop:=lt_of_lt_of_le h (min_le_left _ _)
  obtain ⟨c,hc,cl,ch⟩:=covers I stop start (max start (tile*I)) hI hlo hstop
  have hs:=member_shape I stop start hI c hc
  have hb:c.hi ≤ boundary I c.lo:=by rw [hs.2.2.2,next_eq I stop c.lo hI]; exact min_le_left _ _
  have hq:(max start (tile*I))/I=tile:=Nat.div_eq_of_lt_le (le_max_right _ _)
    (lt_of_lt_of_le h (min_le_right _ _))
  exact ⟨c,hc,(same_tile I _ _ _ hI hb cl ch).symm.trans hq⟩

/-- Every nonterminal source chunk produces exactly the reducer predicate. -/
theorem writer_read (q r I program : Nat) (hI : 0 < I) (c : Chunk)
    (hc : c ∈ chunks I (edge q r (program+1)) (edge q r program) hI)
    (hn : c.hi%I≠0) : readsScratch q r I (c.tile I) program := by
  have hs:=member_shape I _ _ hI c hc
  have he:=nonterminal_end I _ _ hI c hc hn
  refine ⟨by omega,?_,?_⟩
  · rwa [←he]
  · exact (scratch_tile I _ _ hI c hc hn).symm

#axiomsClean member_start
#axiomsClean intersection
#axiomsClean tile_chunk_unique
#axiomsClean intersection_emitted
#axiomsClean writer_read
end Protocol
end VeriTile.Bench.Repairs.StreamK.Whole001
