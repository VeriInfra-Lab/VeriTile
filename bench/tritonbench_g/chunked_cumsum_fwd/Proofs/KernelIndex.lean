import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelGeometry

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton
open Math

/-- Metadata bounds already imposed by the selected checked source. -/
structure Sizes (c : Geometry) : Prop where
  batches : c.batches < 2^63
  sequence : c.sequence < 2^63
  heads : c.heads < 2^63
  chunk : c.chunk < 2^63

theorem start_lt {n d i : Nat} (hd : 0 < d) (hi : i < ceilDiv n d) : i*d < n := by
  have h := (Nat.lt_div_iff_mul_lt hd).1 hi
  unfold ceilDiv at hi
  omega

theorem chunk_start_lt (c : Geometry) (l : Launch) (hl : c.valid l) :
    l.chunk*c.chunk < c.sequence := start_lt c.chunkPositive hl.2.1

theorem headBlock_le (c : Geometry) : c.BH ≤ 64 := by
  have h := c.legalHeadBlock
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h <;> omega

theorem column_bound (c : Geometry) (s : Sizes c) (j : Fin c.BC) : j.val < 2^64 := by
  have hc : c.BC ≤ 2^63 := Nat.pow_le_pow_right (by decide)
    (Nat.clog_le_of_le_pow (Nat.le_of_lt s.chunk))
  have hj := j.isLt
  omega

theorem row_bound (c : Geometry) (i : Fin c.BH) : i.val < 2^64 := by
  have hb := headBlock_le c
  have hi := i.isLt
  omega

theorem launch_bounds (c : Geometry) (s : Sizes c) (l : Launch) (hl : c.valid l) :
    l.batch < 2^64 ∧ l.chunk < 2^64 ∧ l.head < 2^64 := by
  have hb := s.batches
  have hbatch := hl.1
  have hs := s.sequence
  have hh := s.heads
  have hk := start_lt c.chunkPositive hl.2.1
  have hhead := start_lt c.headPositive hl.2.2
  have hc := c.chunkPositive
  have hhpos := c.headPositive
  have hkm : l.chunk ≤ l.chunk*c.chunk := Nat.le_mul_of_pos_right _ hc
  have hhm : l.head ≤ l.head*c.BH := Nat.le_mul_of_pos_right _ hhpos
  omega

theorem scalar_mods (c : Geometry) (s : Sizes c) (l : Launch) (hl : c.valid l) :
    c.sequence % 2^64 = c.sequence ∧ c.heads % 2^64 = c.heads ∧
    c.chunk % 2^64 = c.chunk ∧ l.batch % 2^64 = l.batch ∧
    l.chunk % 2^64 = l.chunk ∧ l.head % 2^64 = l.head := by
  have hs := s.sequence
  have hh := s.heads
  have hc := s.chunk
  have hp := launch_bounds c s l hl
  exact ⟨Nat.mod_eq_of_lt (by omega),Nat.mod_eq_of_lt (by omega),
    Nat.mod_eq_of_lt (by omega),Nat.mod_eq_of_lt hp.1,
    Nat.mod_eq_of_lt hp.2.1,Nat.mod_eq_of_lt hp.2.2⟩

theorem row_mod (c : Geometry) (i : Fin c.BH) : i.val % 2^64 = i.val :=
  Nat.mod_eq_of_lt (row_bound c i)
theorem column_mod (c : Geometry) (s : Sizes c) (j : Fin c.BC) : j.val % 2^64 = j.val :=
  Nat.mod_eq_of_lt (column_bound c s j)

namespace Index
abbrev u (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
def head (c : Geometry) (l : Launch) (i : Nat) := u l.head*u c.BH+u i
def remaining (c : Geometry) (l : Launch) := u c.sequence-u l.chunk*u c.chunk
def dt (c : Geometry) (l : Launch) (i j : Nat) :=
  u l.batch*u c.dtBatch+u l.chunk*u c.chunk*u c.dtTime+
    (head c l i*u c.dtHead+u j*u c.dtTime)
def rate (c : Geometry) (l : Launch) (i : Nat) := head c l i*u c.rateHead
def bias (c : Geometry) (l : Launch) (i : Nat) := head c l i*u c.biasHead
def out (c : Geometry) (l : Launch) (i j : Nat) :=
  u l.batch*u c.layout.outBatch+u l.chunk*u c.layout.outChunk+
    (head c l i*u c.layout.outHead+u j*u c.layout.outColumn)
def address (dataPtr : Nat) (off : BitVec 64) := u dataPtr+off*u 4
end Index

theorem head_word (c : Geometry) (l : Launch) (i : Nat) :
    Index.head c l i = Index.u (Math.head c.layout l i) := by
  simp only [Index.head,Index.u,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  rfl
theorem dt_word (c : Geometry) (l : Launch) (i j : Nat) :
    Index.dt c l i j = Index.u (Math.dtAddress c.layout l i j) := by
  simp only [Index.dt,head_word,Index.u,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  congr 1
  simp [Math.dtAddress,Geometry.layout,Nat.add_assoc]
theorem out_word (c : Geometry) (l : Launch) (i j : Nat) :
    Index.out c l i j = Index.u (Math.outAddress c.layout l i j) := by
  simp only [Index.out,head_word,Index.u,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  congr 1
  simp [Math.outAddress,Nat.add_assoc]

theorem head_bound (c : Geometry) (s : Sizes c) (l : Launch) (hl : c.valid l)
    (i : Fin c.BH) : Math.head c.layout l i.val < 2^64 := by
  have hb := start_lt c.headPositive hl.2.2
  have hBH := headBlock_le c
  have hh := s.heads
  have hi := i.isLt
  change l.head*c.BH+i.val < _
  omega

theorem remaining_exact (c : Geometry) (s : Sizes c) (l : Launch) (hl : c.valid l) :
    (Index.remaining c l).toNat = c.sequence-l.chunk*c.chunk := by
  have hs := chunk_start_lt c l hl
  have hn : l.chunk*c.chunk < 2^64 := by have h := s.sequence; omega
  have hr : c.sequence-l.chunk*c.chunk < 2^64 := by have h := s.sequence; omega
  simp only [Index.remaining,Index.u,← BitVec.ofNat_mul]
  rw [BitVec.ofNat_sub_ofNat_of_le _ _ hn (by omega)]
  exact Nat.mod_eq_of_lt hr

theorem masks (c : Geometry) (s : Sizes c) (l : Launch) (hl : c.valid l)
    (i : Fin c.BH) (j : Fin c.BC) :
    (Index.head c l i.val).ult (Index.u c.heads) = decide (Math.head c.layout l i.val<c.heads) ∧
    (Index.u j.val).ult (Index.u c.chunk) = decide (j.val<c.chunk) ∧
    ((Index.u j.val).toNat < min (Index.u c.chunk).toNat (Index.remaining c l).toNat ↔
      j.val < min c.chunk (c.sequence-l.chunk*c.chunk)) := by
  have hh : (Index.head c l i.val).toNat = Math.head c.layout l i.val := by
    rw [head_word]; exact Nat.mod_eq_of_lt (head_bound c s l hl i)
  have hj : (Index.u j.val).toNat=j.val := Nat.mod_eq_of_lt (column_bound c s j)
  have hheads : (Index.u c.heads).toNat=c.heads := Nat.mod_eq_of_lt (by have h := s.heads; omega)
  have hchunk : (Index.u c.chunk).toNat=c.chunk := Nat.mod_eq_of_lt (by have h := s.chunk; omega)
  simp only [BitVec.ult_eq_decide,hh,hj,hheads,hchunk,remaining_exact c s l hl]
  simp

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
