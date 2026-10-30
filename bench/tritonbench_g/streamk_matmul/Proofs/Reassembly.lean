import bench.tritonbench_g.streamk_matmul.Proofs.Arithmetic
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Reassembly
open Source Protocol Plan Arithmetic
set_option maxHeartbeats 16000000
noncomputable section
local instance (q r I tile program : Nat) : Decidable (readsScratch q r I tile program) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def edgeAt (p : Parameters) (pr : Nat) :=
  edge ((select p.h).streamTiles*(select p.h).iterations/p.h.programs)
    ((select p.h).streamTiles*(select p.h).iterations%p.h.programs) pr

def clipped (p : Parameters) (tile x : Nat) :=
  min ((tile+1)*(select p.h).iterations) (max (tile*(select p.h).iterations) x)

def contribution (p : Parameters) (f : Nat → ℝ) (tile pr : Nat) :=
  sum f (clipped p tile (edgeAt p (pr+1)))-sum f (clipped p tile (edgeAt p pr))

def terminalDelta (p : Parameters) (f : Nat → ℝ) (tile : Nat) (ht : tile < (select p.h).streamTiles) :=
  let t:=Reference.terminal p tile ht
  sum f t.chunk.hi-sum f t.chunk.lo

def partialDelta (p : Parameters) (f : Nat → ℝ) (tile : Nat) (ht : tile < (select p.h).streamTiles)
    (pr : Fin p.h.programs) :=
  if hr:readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val then
    let c:=(Reference.partialChunk p tile ht pr hr).chunk
    sum f c.hi-sum f c.lo
  else 0

theorem chunk_contribution (p : Parameters) (f : Nat → ℝ) (tile : Nat)
    (ht : tile < (select p.h).streamTiles) (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val (Reference.positiveI p tile ht))
    (he : c.tile (select p.h).iterations=tile) :
    contribution p f tile pr.val=sum f c.hi-sum f c.lo := by
  have hs:=member_shape _ _ _ (Reference.positiveI p tile ht) c hc
  have hi:=intersection _ _ _ (Reference.positiveI p tile ht) c hc
  rw [he] at hi
  have hl:clipped p tile (edgeAt p pr.val)=c.lo:=by
    unfold clipped edgeAt
    omega
  have hh:clipped p tile (edgeAt p (pr.val+1))=c.hi:=by
    unfold clipped edgeAt
    omega
  unfold contribution
  rw [hl,hh]

theorem terminal_no_read (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles) :
    ¬readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile
      (Reference.terminal p tile ht).program.val := by
  intro hr
  let t:=Reference.terminal p tile ht
  let c:=Reference.partialChunk p tile ht t.program hr
  have eq:=tile_chunk_unique _ _ _ (Reference.positiveI p tile ht) t.chunk c.chunk t.occurs c.occurs
    (t.tile_eq.trans c.tile_eq.symm)
  have hn:=c.nonterminal
  rw [←eq] at hn
  exact hn t.terminal

theorem inactive_intersection (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles)
    (pr : Fin p.h.programs) (hp : pr≠(Reference.terminal p tile ht).program)
    (hr : ¬readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val) :
    ¬max (edgeAt p pr.val) (tile*(select p.h).iterations) <
      min (edgeAt p (pr.val+1)) ((tile+1)*(select p.h).iterations) := by
  intro hh
  obtain ⟨c,hc,he⟩:=intersection_emitted _ _ _ tile (Reference.positiveI p tile ht) hh
  by_cases hn:c.hi%(select p.h).iterations=0
  · let t:=Reference.terminal p tile ht
    have unique:=initializer_unique (select p.h).streamTiles (select p.h).iterations p.h.programs
      tile pr.val t.program.val (Reference.positiveI p tile ht) c t.chunk
      ⟨pr.isLt,hc,he,hn⟩ ⟨t.program.isLt,t.occurs,t.tile_eq,t.terminal⟩
    exact hp (Fin.ext unique.1)
  · have read:=writer_read _ _ _ pr.val (Reference.positiveI p tile ht) c hc hn
    rw [he] at read
    have hdiv:=selected_divmod p.h (show 0 < (select p.h).streamTiles by omega)
    apply hr
    rw [hdiv.1,hdiv.2]
    exact read

theorem contribution_decomposition (p : Parameters) (f : Nat → ℝ) (tile : Nat)
    (ht : tile < (select p.h).streamTiles) (pr : Fin p.h.programs) :
    contribution p f tile pr.val=
      (if pr=(Reference.terminal p tile ht).program then terminalDelta p f tile ht else 0)+
      partialDelta p f tile ht pr := by
  by_cases hp:pr=(Reference.terminal p tile ht).program
  · subst pr
    rw [if_pos rfl]
    have ht':=terminal_no_read p tile ht
    simp only [partialDelta,dif_neg ht',add_zero]
    let t:=Reference.terminal p tile ht
    exact chunk_contribution p f tile ht t.program t.chunk t.occurs t.tile_eq
  · rw [if_neg hp,zero_add]
    unfold partialDelta
    split
    next hr =>
      let c:=Reference.partialChunk p tile ht pr hr
      exact chunk_contribution p f tile ht pr c.chunk c.occurs c.tile_eq
    next hr =>
      have hx:=inactive_intersection p tile ht pr hp hr
      have hm:=edge_mono ((select p.h).streamTiles*(select p.h).iterations/p.h.programs)
        ((select p.h).streamTiles*(select p.h).iterations%p.h.programs) pr.val (pr.val+1) (by omega)
      have hb:tile*(select p.h).iterations ≤ (tile+1)*(select p.h).iterations:=Nat.mul_le_mul_right _ (by omega)
      have he:clipped p tile (edgeAt p pr.val)=clipped p tile (edgeAt p (pr.val+1)):=by
        unfold clipped edgeAt at *
        omega
      simp [contribution,he]

theorem telescopes (F : Nat → ℝ) (n : Nat) :
    sum (fun i=>F (i+1)-F i) n=F n-F 0 := by
  induction n with
  | zero => simp [sum]
  | succ n ih => rw [sum,ih]; ring

theorem all_contributions (p : Parameters) (f : Nat → ℝ) (tile : Nat)
    (ht : tile < (select p.h).streamTiles) :
    (∑ pr : Fin p.h.programs,contribution p f tile pr.val)=
      sum f ((tile+1)*(select p.h).iterations)-sum f (tile*(select p.h).iterations) := by
  rw [sum_fin]
  unfold contribution
  rw [telescopes (fun pr=>sum f (clipped p tile (edgeAt p pr))) p.h.programs]
  have hP:=(selected_active p.h (show 0 < (select p.h).streamTiles by omega)).1
  have hlast:=edge_last ((select p.h).streamTiles*(select p.h).iterations) p.h.programs hP
  have hbound:=Nat.mul_le_mul_right (select p.h).iterations (show tile+1 ≤ (select p.h).streamTiles by omega)
  have hbase:=Nat.mul_le_mul_right (select p.h).iterations (show tile ≤ tile+1 by omega)
  have hl:clipped p tile (edgeAt p p.h.programs)=(tile+1)*(select p.h).iterations:=by
    unfold clipped edgeAt
    rw [hlast]
    omega
  have hz:clipped p tile (edgeAt p 0)=tile*(select p.h).iterations:=by
    unfold clipped edgeAt
    rw [edge_zero]
    omega
  rw [hl,hz]

theorem deltas_total (p : Parameters) (f : Nat → ℝ) (tile : Nat)
    (ht : tile < (select p.h).streamTiles) :
    terminalDelta p f tile ht+(∑ pr : Fin p.h.programs,partialDelta p f tile ht pr)=
      sum f ((tile+1)*(select p.h).iterations)-sum f (tile*(select p.h).iterations) := by
  have hd: (∑ pr : Fin p.h.programs,contribution p f tile pr.val)=
      terminalDelta p f tile ht+(∑ pr : Fin p.h.programs,partialDelta p f tile ht pr) := by
    simp_rw [contribution_decomposition p f tile ht]
    rw [Finset.sum_add_distrib]
    simp
  rw [←hd]
  exact all_contributions p f tile ht

/-- Stream assembly counts every exact-Real K-block once. Only this ideal
arithmetic theorem reassociates addition; the FP result remains pid-ordered. -/
theorem stream_deltas (p : Parameters) (a : A p → ℝ) (b : B p → ℝ) (tile : Nat)
    (ht : tile < (select p.h).streamTiles) (lane : Lane p) :
    Reference.streamValue p real a b tile ht lane=
      terminalDelta p (iteration p a b tile lane) tile ht+
        ∑ pr : Fin p.h.programs,partialDelta p (iteration p a b tile lane) tile ht pr := by
  let t:=Reference.terminal p tile ht
  have ts:=member_shape _ _ _ (Reference.positiveI p tile ht) t.chunk t.occurs
  have hi:=chunk_real p a b tile t.chunk.lo t.chunk.hi (by omega) lane
  unfold Reference.streamValue
  dsimp only
  rw [hi]
  have heq:(fun value pr=>
      if hr:readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val then
        let c:=(Reference.partialChunk p tile ht pr hr).chunk
        real.atomicAdd value (chunkValue p real a b tile c.lo c.hi lane)
      else value)=
      (fun value pr=>value+partialDelta p (iteration p a b tile lane) tile ht pr) := by
    funext value pr
    unfold partialDelta
    split
    next hr =>
      let c:=Reference.partialChunk p tile ht pr hr
      have cs:=member_shape _ _ _ (Reference.positiveI p tile ht) c.chunk c.occurs
      have hc:=chunk_real p a b tile c.chunk.lo c.chunk.hi (by omega) lane
      change value+chunkValue p real a b tile c.chunk.lo c.chunk.hi lane=_
      rw [hc]
    next hr => simp [real]
  rw [heq,fin_fold]
  rfl

theorem periodic_total (p : Parameters) (a : A p → ℝ) (b : B p → ℝ) (tile : Nat)
    (lane : Lane p) :
    sum (iteration p a b tile lane) ((tile+1)*(select p.h).iterations)-
      sum (iteration p a b tile lane) (tile*(select p.h).iterations)=
      chunkValue p real a b tile 0 (select p.h).iterations lane := by
  rw [Nat.add_mul,Nat.one_mul,sum_add]
  rw [chunk_real p a b tile 0 _ (Nat.zero_le _) lane]
  simp only [sum,sub_zero,add_sub_cancel_left]
  apply sum_congr
  intro k hk
  simp only [iteration,Nat.add_mod,Nat.mul_mod_left,Nat.zero_add,Nat.mod_eq_of_lt hk]

/-- Independent matrix-product closure for the stream prefix. -/
theorem stream_real (p : Parameters) (a : A p → ℝ) (b : B p → ℝ) (tile : Nat)
    (ht : tile < (select p.h).streamTiles) (lane : Lane p) (ha : active p tile lane) :
    Reference.streamValue p real a b tile ht lane=
      Reference.matrixProduct p a b ⟨row p tile lane,ha.1⟩ ⟨col p tile lane,ha.2⟩ := by
  rw [stream_deltas,deltas_total,periodic_total,full_real p a b tile lane ha]

#axiomsClean contribution_decomposition
#axiomsClean all_contributions
#axiomsClean stream_deltas
#axiomsClean stream_real
end
end Reassembly
end VeriTile.Bench.Repairs.StreamK.Whole001
