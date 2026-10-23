import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Storage
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeIndex

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Native
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open Storage

/-- Standard Torch dimension metadata after successful allocation; no signed32
or stride-product assumption. Unsigned arithmetic may wrap on masked lanes. -/
structure Sizes (c : Host) : Prop where
  sequence : c.sequence < 2^63
  heads : c.heads < 2^63
  chunk : c.chunk < 2^63

theorem u_exact (x : Nat) (h : x < 2^64) : (Index.u x).toNat = x := Nat.mod_eq_of_lt h

theorem start_lt {n d i : Nat} (hd : 0 < d) (hi : i < Whole.ceilDiv n d) : i*d < n := by
  have h := (Nat.lt_div_iff_mul_lt hd).1 hi
  unfold Whole.ceilDiv at hi
  omega

theorem head_bound (c : Host) (s : Sizes c) (l : Launch) (hl : c.valid l)
    (i : Fin c.BH) : Math.head c.layout l i.val < 2^64 := by
  have hb := start_lt c.headPositive hl.2.2
  have hBH : c.BH ≤ 64 := by
    have h := c.legalHeadBlock
    simp only [List.mem_cons,List.not_mem_nil,or_false] at h
    rcases h with h|h|h|h|h|h|h <;> omega
  have hh := s.heads
  have hi := i.isLt
  change l.head*c.BH+i.val < _
  omega

theorem column_bound (c : Host) (s : Sizes c) (j : Fin c.BC) : j.val < 2^64 := by
  have hc : c.BC ≤ 2^63 := Nat.pow_le_pow_right (by decide)
    (Nat.clog_le_of_le_pow (Nat.le_of_lt s.chunk))
  have hj := j.isLt
  omega

theorem head_word (c : Host) (l : Launch) (i : Nat) :
    Index.head c l i = Index.u (Math.head c.layout l i) := by
  simp only [Index.head,Index.u,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  rfl

theorem dt_word (c : Host) (l : Launch) (i j : Nat) :
    Index.dt c l i j = Index.u (Math.dtAddress c.layout l i j) := by
  simp only [Index.dt,head_word,Index.u,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  congr 1
  simp [Math.dtAddress,Whole.Host.layout,Nat.add_assoc]

theorem out_word (c : Host) (l : Launch) (i j : Nat) :
    Index.out c l i j = Index.u (Math.outAddress c.layout l i j) := by
  simp only [Index.out,head_word,Index.u,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  congr 1
  simp [Math.outAddress,Nat.add_assoc]

theorem rate_word (c : Host) (l : Launch) (i : Nat) :
    Index.rate c l i = Index.u (Math.head c.layout l i*c.rateHead) := by
  simp only [Index.rate,head_word,Index.u,← BitVec.ofNat_mul]

theorem bias_word (c : Host) (l : Launch) (i : Nat) :
    Index.bias c l i = Index.u (Math.head c.layout l i*c.biasHead) := by
  simp only [Index.bias,head_word,Index.u,← BitVec.ofNat_mul]

theorem remaining_exact (c : Host) (s : Sizes c) (l : Launch) (hl : c.valid l) :
    (Index.remaining c l).toNat = c.sequence-l.chunk*c.chunk := by
  have hs := Whole.chunk_start_lt c l hl
  have hn : l.chunk*c.chunk < 2^64 := by have h := s.sequence; omega
  have hr : c.sequence-l.chunk*c.chunk < 2^64 := by have h := s.sequence; omega
  simp only [Index.remaining,Index.u,← BitVec.ofNat_mul]
  rw [BitVec.ofNat_sub_ofNat_of_le _ _ hn (by omega)]
  exact Nat.mod_eq_of_lt hr

theorem masks (c : Host) (s : Sizes c) (l : Launch) (hl : c.valid l)
    (i : Fin c.BH) (j : Fin c.BC) :
    (Index.head c l i.val).ult (Index.u c.heads) =
      decide (Math.head c.layout l i.val < c.heads) ∧
    (Index.u j.val).ult (Index.u c.chunk) = decide (j.val < c.chunk) ∧
    ((Index.u j.val).toNat < min (Index.u c.chunk).toNat (Index.remaining c l).toNat ↔
      j.val < min c.chunk (c.sequence-l.chunk*c.chunk)) := by
  have hh : (Index.head c l i.val).toNat = Math.head c.layout l i.val := by
    rw [head_word]; exact u_exact _ (head_bound c s l hl i)
  have hj := u_exact j.val (column_bound c s j)
  have hheads := u_exact c.heads (show c.heads < 2^64 by have h := s.heads; omega)
  have hchunk := u_exact c.chunk (show c.chunk < 2^64 by have h := s.chunk; omega)
  simp only [BitVec.ult_eq_decide,hh,hj,hheads,hchunk,remaining_exact c s l hl]
  simp

/-- Native byte addresses are derived from the actual same-call descriptor;
no unrelated backing or hidden product-range premise is used. -/
theorem active_dt (c : Host) (l : Launch) (i j : Nat) (t : Tensor alloc)
    (covered : t.Covers (Math.dtAddress c.layout l i j)) :
    (Index.address t.dataPtr (Index.dt c l i j)).toNat =
      t.dataPtr+4*Math.dtAddress c.layout l i j ∧
    t.dataPtr+4*Math.dtAddress c.layout l i j+4 ≤
      (alloc t.view.allocation).address+(alloc t.view.allocation).bytes ∧
    t.dataPtr+4*Math.dtAddress c.layout l i j+4 ≤ 2^64 := by
  rw [dt_word]
  exact ⟨native_address t _ covered,span t _ covered⟩

theorem active_out (c : Host) (l : Launch) (i j : Nat) (t : Tensor alloc)
    (covered : t.Covers (Math.outAddress c.layout l i j)) :
    (Index.address t.dataPtr (Index.out c l i j)).toNat =
      t.dataPtr+4*Math.outAddress c.layout l i j ∧
    t.dataPtr+4*Math.outAddress c.layout l i j+4 ≤
      (alloc t.view.allocation).address+(alloc t.view.allocation).bytes ∧
    t.dataPtr+4*Math.outAddress c.layout l i j+4 ≤ 2^64 := by
  rw [out_word]
  exact ⟨native_address t _ covered,span t _ covered⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Native
