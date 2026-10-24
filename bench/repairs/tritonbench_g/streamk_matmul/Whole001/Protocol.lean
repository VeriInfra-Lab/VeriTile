import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Plan
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Protocol
open Plan
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

theorem boundary_eq (I current : Nat) (hI : 0 < I) :
    current+(I-current%I)=boundary I current := by
  have hm:=Nat.mod_lt current hI
  have hd:=Nat.div_add_mod current I
  simp only [boundary,Nat.add_mul,Nat.one_mul]
  rw [Nat.mul_comm (current/I) I]
  omega

theorem next_eq (I stop current : Nat) (hI : 0 < I) :
    next I stop current=min (boundary I current) stop := by
  rw [next,boundary_eq I current hI]

theorem boundary_progress (I current : Nat) (hI : 0 < I) :
    current < boundary I current := by
  rw [←boundary_eq I current hI]
  have hm:=Nat.mod_lt current hI
  omega

theorem same_tile (I lo hi i : Nat) (hI : 0 < I)
    (hh : hi ≤ boundary I lo) (hl : lo ≤ i) (hr : i < hi) : i/I=lo/I := by
  have hmono : lo/I ≤ i/I := Nat.div_le_div_right hl
  have bound : i/I < lo/I+1 := (Nat.div_lt_iff_lt_mul hI).mpr (by exact lt_of_lt_of_le hr hh)
  omega

/-- Exact source while-loop, structurally terminating because each source next
boundary strictly increases. No scheduler, lock, or completion premise. -/
@[simp] theorem chunks_nil (I stop start : Nat) (hI : 0 < I) (h : stop ≤ start) :
    chunks I stop start hI=[] := by rw [chunks]; simp [show ¬start < stop by omega]

theorem chunks_cons (I stop start : Nat) (hI : 0 < I) (h : start < stop) :
    chunks I stop start hI=⟨start,next I stop start⟩::chunks I stop (next I stop start) hI := by
  rw [chunks]; simp [h]

/-- Every emitted chunk is nonempty, stays inside its program interval, and
stops at the source minimum of the next tile boundary and program end. -/
theorem member_shape (I stop start : Nat) (hI : 0 < I) (c : Chunk)
    (hc : c ∈ chunks I stop start hI) :
    start ≤ c.lo ∧ c.lo < c.hi ∧ c.hi ≤ stop ∧ c.hi=next I stop c.lo := by
  rw [chunks] at hc
  split at hc
  next h =>
    simp only [List.mem_cons] at hc
    rcases hc with rfl|hc
    · have hn:=next_progress I stop start hI h
      exact ⟨le_rfl,hn.1,hn.2,rfl⟩
    · have ih:=member_shape I stop (next I stop start) hI c hc
      have hn:=next_progress I stop start hI h
      exact ⟨by omega,ih.2⟩
  next h => simp at hc
termination_by stop-start
decreasing_by
  have hn:=next_progress I stop start hI (by assumption)
  omega

/-- The actual while-loop covers each program iteration exactly once. -/
theorem covers (I stop start i : Nat) (hI : 0 < I) (hl : start ≤ i) (hr : i < stop) :
    ∃ c ∈ chunks I stop start hI, c.lo ≤ i ∧ i < c.hi := by
  have h:start < stop:=by omega
  rw [chunks_cons I stop start hI h]
  by_cases hi:i < next I stop start
  · exact ⟨⟨start,next I stop start⟩,List.mem_cons_self,hl,hi⟩
  · obtain ⟨c,hc,cl,ch⟩:=covers I stop (next I stop start) i hI (by omega) hr
    exact ⟨c,List.mem_cons_of_mem _ hc,cl,ch⟩
termination_by stop-start
decreasing_by
  have hn:=next_progress I stop start hI (by assumption)
  omega

theorem disjoint (I stop start : Nat) (hI : 0 < I) (c d : Chunk)
    (hc : c ∈ chunks I stop start hI) (hd : d ∈ chunks I stop start hI)
    (i : Nat) (ci : c.lo ≤ i ∧ i < c.hi) (di : d.lo ≤ i ∧ i < d.hi) : c=d := by
  by_cases h:start < stop
  · rw [chunks_cons I stop start hI h] at hc hd
    simp only [List.mem_cons] at hc hd
    rcases hc with rfl|hc
    · rcases hd with rfl|hd
      · rfl
      · have hs:=member_shape I stop (next I stop start) hI d hd
        simp only at ci
        omega
    · rcases hd with rfl|hd
      · have hs:=member_shape I stop (next I stop start) hI c hc
        simp only at di
        omega
      · exact disjoint I stop (next I stop start) hI c d hc hd i ci di
  · rw [chunks_nil I stop start hI (by omega)] at hc
    simp at hc
termination_by stop-start
decreasing_by
  have hn:=next_progress I stop start hI (by assumption)
  omega

theorem nonterminal_end (I stop start : Nat) (hI : 0 < I) (c : Chunk)
    (hc : c ∈ chunks I stop start hI) (hn : c.hi%I≠0) : c.hi=stop := by
  have hs:=member_shape I stop start hI c hc
  rw [hs.2.2.2] at hn ⊢
  exact nonterminal_last I stop c.lo hI hn

/-- Exactly the reducer's first two tests characterize initialized scratch. -/
theorem scratch_initialized (I stop start : Nat) (hI : 0 < I) :
    (∃ c ∈ chunks I stop start hI, c.hi%I≠0) ↔ start < stop ∧ stop%I≠0 := by
  constructor
  · rintro ⟨c,hc,hn⟩
    have hs:=member_shape I stop start hI c hc
    have he:=nonterminal_end I stop start hI c hc hn
    exact ⟨by omega,by rwa [←he]⟩
  · rintro ⟨h,hn⟩
    obtain ⟨c,hc,hl,hr⟩:=covers I stop start (stop-1) hI (by omega) (by omega)
    have hs:=member_shape I stop start hI c hc
    have he:c.hi=stop:=by omega
    exact ⟨c,hc,by rwa [he]⟩

/-- There is only one scratch write per program, hence no overwritten value. -/
theorem scratch_unique (I stop start : Nat) (hI : 0 < I) (c d : Chunk)
    (hc : c ∈ chunks I stop start hI) (hd : d ∈ chunks I stop start hI)
    (hn : c.hi%I≠0) (hn' : d.hi%I≠0) : c=d := by
  have hs:=member_shape I stop start hI c hc
  have ht:=member_shape I stop start hI d hd
  have he:=nonterminal_end I stop start hI c hc hn
  have hf:=nonterminal_end I stop start hI d hd hn'
  exact disjoint I stop start hI c d hc hd (stop-1) (by omega) (by omega)

/-- A nonterminal chunk's tile is the reducer's integer end/I, including when
its interval begins in an earlier tile. -/
theorem scratch_tile (I stop start : Nat) (hI : 0 < I) (c : Chunk)
    (hc : c ∈ chunks I stop start hI) (hn : c.hi%I≠0) : c.tile I=stop/I := by
  have hs:=member_shape I stop start hI c hc
  have he:=nonterminal_end I stop start hI c hc hn
  have hb:stop ≤ boundary I c.lo:=by rw [←he,hs.2.2.2,next_eq I stop c.lo hI]; exact min_le_left _ _
  have hn':stop < boundary I c.lo:=by
    by_contra h
    have eq:stop=boundary I c.lo:=by omega
    rw [he,eq,boundary,Nat.mul_mod_left] at hn
    contradiction
  exact (same_tile I c.lo (boundary I c.lo) stop hI le_rfl (by omega) hn').symm

/-- No scratch read is conditional on a numerical value or publication flag. -/
theorem read_has_unique_writer (q r I tile program : Nat) (hI : 0 < I)
    (hr : readsScratch q r I tile program) :
    ∃! c, c ∈ chunks I (edge q r (program+1)) (edge q r program) hI ∧
      c.hi%I≠0 ∧ c.tile I=tile := by
  obtain ⟨c,hc,hn⟩:=(scratch_initialized I (edge q r (program+1)) (edge q r program) hI).mpr ⟨hr.1,hr.2.1⟩
  refine ⟨c,⟨hc,hn,?_⟩,?_⟩
  · exact (scratch_tile I _ _ hI c hc hn).trans hr.2.2
  · intro d hd
    exact scratch_unique I _ _ hI d c hd.1 hc hd.2.1 hn


theorem terminal_boundary (I stop start : Nat) (hI : 0 < I) (c : Chunk)
    (hc : c ∈ chunks I stop start hI) (ht : c.hi%I=0) :
    c.hi=(c.tile I+1)*I := by
  have hs:=member_shape I stop start hI c hc
  have hh:c.hi ≤ boundary I c.lo:=by
    rw [hs.2.2.2,next_eq I stop c.lo hI]
    exact min_le_left _ _
  change c.hi=boundary I c.lo
  by_contra he
  have hx:c.hi < (c.lo/I+1)*I:=by change c.hi < boundary I c.lo; omega
  have hd:c.hi/I < c.lo/I+1:=(Nat.div_lt_iff_lt_mul hI).mpr hx
  have hd':c.hi/I ≤ c.lo/I:=by omega
  have hm:=Nat.mul_le_mul_right I hd'
  have hdiv:=Nat.div_add_mod c.hi I
  rw [ht,Nat.add_zero,Nat.mul_comm] at hdiv
  have hlo:=Nat.div_mul_le_self c.lo I
  omega

def TerminalWriter (S I P tile p : Nat) (hI : 0 < I) (c : Chunk) : Prop :=
  p < P ∧ c ∈ programChunks S I P p hI ∧ c.tile I=tile ∧ c.hi%I=0

/-- Each stream tile receives one ordinary store in first_wave before the
ordered reduction launch begins. Its initializer is obtained from the actual
while-loop and the unique program interval containing its final iteration. -/
theorem initializer_exists (S I P tile : Nat) (hI : 0 < I) (hP : 0 < P) (ht : tile < S) :
    ∃ p c, TerminalWriter S I P tile p hI c := by
  obtain ⟨p,hp,hu⟩:=terminal_owner S I P tile hI hP ht
  let last:=(tile+1)*I-1
  have hlast:0 < (tile+1)*I:=Nat.mul_pos (by omega) hI
  have split:(tile+1)*I=tile*I+I:=by simp [Nat.add_mul]
  have lastTile:last/I=tile:=Nat.div_eq_of_lt_le (by dsimp [last]; omega) (by dsimp [last]; omega)
  obtain ⟨c,hc,hl,hr⟩:=covers I (edge (S*I/P) (S*I%P) (p+1))
    (edge (S*I/P) (S*I%P) p) last hI hp.2.1 hp.2.2
  have hs:=member_shape I _ _ hI c hc
  have hb:c.hi ≤ boundary I c.lo:=by rw [hs.2.2.2,next_eq I _ c.lo hI]; exact min_le_left _ _
  have ct:c.tile I=tile:=(same_tile I c.lo c.hi last hI hb hl hr).symm.trans lastTile
  have che:c.hi=(tile+1)*I:=by
    change c.hi ≤ (c.tile I+1)*I at hb
    rw [ct] at hb
    dsimp [last] at hr
    omega
  exact ⟨p,c,hp.1,hc,ct,by rw [che,Nat.mul_mod_left]⟩

/-- No schedule can race two initializing stores to the same logical tile. -/
theorem initializer_unique (S I P tile p q : Nat) (hI : 0 < I) (c d : Chunk)
    (hc : TerminalWriter S I P tile p hI c) (hd : TerminalWriter S I P tile q hI d) :
    p=q ∧ c=d := by
  have cs:=member_shape I _ _ hI c hc.2.1
  have ds:=member_shape I _ _ hI d hd.2.1
  have ce:=terminal_boundary I _ _ hI c hc.2.1 hc.2.2.2
  have de:=terminal_boundary I _ _ hI d hd.2.1 hd.2.2.2
  rw [hc.2.2.1] at ce
  rw [hd.2.2.1] at de
  have ci:c.lo ≤ (tile+1)*I-1 ∧ (tile+1)*I-1 < c.hi:=by omega
  have di:d.lo ≤ (tile+1)*I-1 ∧ (tile+1)*I-1 < d.hi:=by omega
  have pq:p=q:=interval_unique (S*I/P) (S*I%P) p q ((tile+1)*I-1) (by omega) (by omega)
  subst q
  exact ⟨rfl,disjoint I _ _ hI c d hc.2.1 hd.2.1 _ ci di⟩

/-- Every K-block in a stream tile belongs to exactly one actual emitted chunk,
including empty program intervals and programs that cross multiple tiles. -/
theorem iteration_has_unique_chunk (S I P tile iter : Nat) (hI : 0 < I)
    (hP : 0 < P) (ht : tile < S) (hi : iter < I) :
    ∃! pc : Nat × Chunk, pc.1 < P ∧ pc.2 ∈ programChunks S I P pc.1 hI ∧
      pc.2.lo ≤ tile*I+iter ∧ tile*I+iter < pc.2.hi ∧ pc.2.tile I=tile := by
  have bound:tile*I+iter < S*I:=by
    have hm:=Nat.mul_le_mul_right I (show tile+1 ≤ S by omega)
    rw [Nat.add_mul,Nat.one_mul] at hm
    omega
  obtain ⟨p,hp,hu⟩:=full_partition (S*I) P (tile*I+iter) hP bound
  obtain ⟨c,hc,hl,hr⟩:=covers I (edge (S*I/P) (S*I%P) (p+1))
    (edge (S*I/P) (S*I%P) p) (tile*I+iter) hI hp.2.1 hp.2.2
  have hs:=member_shape I _ _ hI c hc
  have hb:c.hi ≤ boundary I c.lo:=by rw [hs.2.2.2,next_eq I _ c.lo hI]; exact min_le_left _ _
  have ct:c.tile I=tile:=by
    have hdiv:(tile*I+iter)/I=tile:=Nat.div_eq_of_lt_le (by omega) (by rw [Nat.add_mul,Nat.one_mul]; omega)
    exact (same_tile I _ _ _ hI hb hl hr).symm.trans hdiv
  refine ⟨(p,c),⟨hp.1,hc,hl,hr,ct⟩,?_⟩
  rintro ⟨q,d⟩ ⟨hq,hd,dl,dr,dt⟩
  dsimp only at hq hd dl dr dt
  have ds:=member_shape I _ _ hI d hd
  have pq:q=p:=interval_unique (S*I/P) (S*I%P) q p (tile*I+iter) (by omega) hp.2
  subst q
  have dc:=disjoint I _ _ hI d c hd hc (tile*I+iter) ⟨dl,dr⟩ ⟨hl,hr⟩
  exact Prod.ext rfl dc

#axiomsClean terminal_boundary
#axiomsClean initializer_exists
#axiomsClean initializer_unique
#axiomsClean iteration_has_unique_chunk

#axiomsClean chunks
#axiomsClean covers
#axiomsClean disjoint
#axiomsClean scratch_initialized
#axiomsClean scratch_unique
#axiomsClean scratch_tile
#axiomsClean read_has_unique_writer
end Protocol
end VeriTile.Bench.Repairs.StreamK.Whole001
