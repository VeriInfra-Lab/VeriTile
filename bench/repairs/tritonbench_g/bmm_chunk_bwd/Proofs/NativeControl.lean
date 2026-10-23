import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeIndex
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostContract

namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeControl
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd

def signedLimit : Nat := 2^63

def elements (h : HostCall) := h.batch*h.seqlen*h.groupCount*h.K

/-- Default-FP32 finite controls. The direct boundary derives these from
explicit byte-extent and nonempty-grid admission checks. The historical field
name scratchBytes denotes the direct output byte extent. -/
structure Metadata (h : HostCall) (c : Tuning) : Prop where
  chunkPositive : 0 < h.chunk
  chunks : h.nchunks = (h.seqlen+h.chunk-1)/h.chunk
  scratchBytes : elements h*4 < signedLimit
  doutNumel : h.batch*h.nchunks*h.groupCount*h.chunk*h.chunk < signedLimit
  gridX : (HostContract.grid h c).tiles ≤ 2147483647
  gridY : h.batch ≤ 65535
  gridZ : h.nchunks*h.groupCount ≤ 65535

private theorem ceil_zero (B : Nat) (hB : 0 < B) : (0+B-1)/B = 0 := by
  apply Nat.div_eq_of_lt
  omega

private theorem before_ceil (N B i : Nat) (hB : 0 < B) (hi : i < (N+B-1)/B) : i*B < N := by
  have h1 : (i+1)*B ≤ ((N+B-1)/B)*B := Nat.mul_le_mul_right B (by omega)
  have h2 := Nat.div_mul_le_self (N+B-1) B
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

private theorem padded_bound (N B tile lane : Nat) (hB : 0 < B)
    (ht : tile < (N+B-1)/B) (hl : lane < B) : tile*B+lane < N+B := by
  have := before_ceil N B tile hB ht
  omega

theorem tuning_bounds (choice : Fin 9) :
    0 < (tuning choice).BM ∧ (tuning choice).BM ≤ 128 ∧
    0 < (tuning choice).BN ∧ (tuning choice).BN ≤ 256 ∧
    0 < (tuning choice).BCS ∧ (tuning choice).BCS ≤ 64 := by
  fin_cases choice <;> decide

/-- Every actually scheduled program is nonempty in all logical dimensions.
Zero-sized API cases are still covered by the empty source grid. -/
theorem nonempty (h : HostCall) (choice : Fin 9) (metadata : Metadata h (tuning choice))
    (q : HostGrid.Pid) (hq : q ∈ HostGrid.programs (HostContract.grid h (tuning choice))) :
    0 < h.batch ∧ 0 < h.seqlen ∧ 0 < h.groupCount ∧ 0 < h.K ∧ 0 < h.nchunks := by
  obtain ⟨ht,hb,hg⟩ := (HostGrid.mem_programs _ _).mp hq
  change q.batch < h.batch at hb
  change q.chunkHead < h.nchunks*h.groupCount at hg
  have hprod : 0 < h.nchunks*h.groupCount := by omega
  have hn : 0 < h.nchunks := by nlinarith
  have hG : 0 < h.groupCount := by nlinarith
  have hs : 0 < h.seqlen := by
    by_contra he
    have hz : h.seqlen = 0 := by omega
    rw [metadata.chunks,hz,ceil_zero _ metadata.chunkPositive] at hn
    omega
  have hk : 0 < h.K := by
    by_contra he
    have hz : h.K = 0 := by omega
    change q.tile < ((h.chunk+(tuning choice).BM-1)/(tuning choice).BM)*
      ((h.K+(tuning choice).BN-1)/(tuning choice).BN) at ht
    rw [hz,ceil_zero _ (tuning_bounds choice).2.2.1,Nat.mul_zero] at ht
    omega
  exact ⟨by omega,hs,hG,hk,hn⟩

/-- Shape magnitudes follow from successful tensor allocation/metadata, not
from newly imposed native-safe conditions. -/
theorem sizes (h : HostCall) (choice : Fin 9) (metadata : Metadata h (tuning choice))
    (q : HostGrid.Pid) (hq : q ∈ HostGrid.programs (HostContract.grid h (tuning choice))) :
    h.seqlen < 2^61 ∧ h.K < 2^61 ∧ h.chunk < 2^32 ∧ h.groupCount ≤ 65535 := by
  obtain ⟨hb,hs,hg,hk,hn⟩ := nonempty h choice metadata q hq
  have he : elements h < 2^61 := by
    have hh := metadata.scratchBytes
    norm_num [signedLimit] at hh ⊢
    omega
  have hkle : h.K ≤ elements h := by
    have hp : 1 ≤ h.batch*h.seqlen*h.groupCount := by
      exact Nat.succ_le_of_lt (Nat.mul_pos (Nat.mul_pos hb hs) hg)
    exact (Nat.one_mul h.K).symm ▸ Nat.mul_le_mul_right h.K hp
  have hsle : h.seqlen ≤ elements h := by
    have hp : 1 ≤ h.batch*h.groupCount*h.K := by
      exact Nat.succ_le_of_lt (Nat.mul_pos (Nat.mul_pos hb hg) hk)
    simpa [elements,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_le_mul_right h.seqlen hp
  have hcc : h.chunk*h.chunk < signedLimit := by
    have hp : 1 ≤ h.batch*h.nchunks*h.groupCount :=
      Nat.succ_le_of_lt (Nat.mul_pos (Nat.mul_pos hb hn) hg)
    have hl := Nat.mul_le_mul_right (h.chunk*h.chunk) hp
    simp only [Nat.one_mul] at hl
    exact lt_of_le_of_lt (by simpa [Nat.mul_assoc] using hl) metadata.doutNumel
  have hc : h.chunk < 2^32 := by
    by_contra hnot
    have hl : 2^32 ≤ h.chunk := by omega
    have square := Nat.mul_le_mul hl hl
    norm_num [signedLimit] at hcc
    norm_num at square
    omega
  have hg' : h.groupCount ≤ h.nchunks*h.groupCount := by
    simpa using Nat.mul_le_mul_right h.groupCount (show 1 ≤ h.nchunks by omega)
  exact ⟨lt_of_le_of_lt hsle he,lt_of_le_of_lt hkle he,hc,le_trans hg' metadata.gridZ⟩

/-- All padded row/feature indices and both ceiling numerators remain signed
64-bit values. This includes inactive lanes rather than silently assuming they
are in range of the logical tensor. -/
theorem lanes (h : HostCall) (choice : Fin 9) (metadata : Metadata h (tuning choice))
    (q : HostGrid.Pid) (hq : q ∈ HostGrid.programs (HostContract.grid h (tuning choice))) :
    let c := tuning choice
    let p := HostContract.parameters h
    h.K+c.BN-1 < signedLimit ∧ h.chunk+c.BM-1 < signedLimit ∧
      (∀ i : Fin c.BM, RealViews.row c.BM c.BN p q.tile i < signedLimit) ∧
      (∀ j : Fin c.BN, RealViews.column c.BN p q.tile j < signedLimit) := by
  dsimp only
  let c := tuning choice
  have tb := tuning_bounds choice
  obtain ⟨_,hK,hC,_⟩ := sizes h choice metadata q hq
  have hNE := nonempty h choice metadata q hq
  let cols := (h.K+c.BN-1)/c.BN
  have colsPos : 0 < cols := by
    apply Nat.div_pos
    · dsimp [cols]; omega
    · exact tb.2.2.1
  have ht := ((HostGrid.mem_programs _ _).mp hq).1
  change q.tile < ((h.chunk+c.BM-1)/c.BM)*cols at ht
  have htile : q.tile/cols < (h.chunk+c.BM-1)/c.BM := (Nat.div_lt_iff_lt_mul colsPos).mpr ht
  have hcol : q.tile%cols < cols := Nat.mod_lt _ colsPos
  refine ⟨?_,?_,?_,?_⟩
  · norm_num [signedLimit] at hK ⊢; have := tb.2.2.2.1; omega
  · norm_num [signedLimit] at hC ⊢; have := tb.2.1; omega
  · intro i
    have hh := padded_bound h.chunk c.BM (q.tile/cols) i.val tb.1 htile i.isLt
    change q.tile/cols*c.BM+i.val < signedLimit
    norm_num [signedLimit] at hC ⊢
    have : c.BM ≤ 128 := tb.2.1
    omega
  · intro j
    have hh := padded_bound h.K c.BN (q.tile%cols) j.val tb.2.2.1 hcol j.isLt
    change q.tile%cols*c.BN+j.val < signedLimit
    norm_num [signedLimit] at hK ⊢
    have : c.BN ≤ 256 := tb.2.2.2.1
    omega

/-- Chunk starts and every executed contraction step are below their logical
limits; signed subtraction therefore agrees with mathematical subtraction. -/
theorem chunk_and_loop (h : HostCall) (choice : Fin 9) (metadata : Metadata h (tuning choice))
    (q : HostGrid.Pid) (hq : q ∈ HostGrid.programs (HostContract.grid h (tuning choice))) :
    let c := tuning choice
    let chunkId := q.chunkHead/h.groupCount
    let length := min h.chunk (h.seqlen-chunkId*h.chunk)
    chunkId*h.chunk < h.seqlen ∧ length+c.BCS-1 < signedLimit ∧
      ∀ step, step < (length+c.BCS-1)/c.BCS →
        step*c.BCS < length ∧ length-step*c.BCS < signedLimit := by
  dsimp only
  have tb := tuning_bounds choice
  obtain ⟨_,_,hg,_,_⟩ := nonempty h choice metadata q hq
  have hch := ((HostGrid.mem_programs _ _).mp hq).2.2
  change q.chunkHead < h.nchunks*h.groupCount at hch
  have hchunk : q.chunkHead/h.groupCount < h.nchunks := (Nat.div_lt_iff_lt_mul hg).mpr hch
  rw [metadata.chunks] at hchunk
  have hstart := before_ceil h.seqlen h.chunk _ metadata.chunkPositive hchunk
  obtain ⟨_,_,hC,_⟩ := sizes h choice metadata q hq
  have hlen := Nat.min_le_left h.chunk (h.seqlen-q.chunkHead/h.groupCount*h.chunk)
  refine ⟨hstart,?_,?_⟩
  · norm_num [signedLimit] at hC ⊢
    have := tb.2.2.2.2.2
    omega
  · intro step hs
    have hb := before_ceil _ (tuning choice).BCS step tb.2.2.2.2.1 hs
    refine ⟨hb,?_⟩
    norm_num [signedLimit] at hC ⊢
    omega

#axiomsClean nonempty
#axiomsClean sizes
#axiomsClean lanes
#axiomsClean chunk_and_loop
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeControl
