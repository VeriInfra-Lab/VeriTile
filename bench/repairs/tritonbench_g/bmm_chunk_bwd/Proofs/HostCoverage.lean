import bench.repairs.tritonbench_g.bmm_chunk_bwd.Source
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostContract

/-! Grid ownership is derived from the exact host cdiv grid. Each valid
chunk/head/output coordinate has a concrete active program and lane. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostCoverage
open VeriTile.Triton
noncomputable section

private theorem ceil_cover (N B : Nat) (hB : 0 < B) : N ≤ (N + B - 1) / B * B := by
  have h := Nat.div_add_mod (N + B - 1) B
  have hm := Nat.mod_lt (N + B - 1) hB
  rw [Nat.mul_comm B] at h
  omega

private theorem quotient_bound (N B x : Nat) (hB : 0 < B) (hx : x < N) :
    x / B < (N + B - 1) / B := by
  apply (Nat.div_lt_iff_lt_mul hB).mpr
  exact lt_of_lt_of_le hx (ceil_cover N B hB)

/-- Every logical active coordinate is covered. Dout remains a full C×C
allocation, while stores use the shortened final-chunk length. -/
theorem covers (BM BN batches nchunks : Nat) (p : RealProgram.Parameters)
    (hBM : 0 < BM) (hBN : 0 < BN) (hg : 0 < p.groups)
    (b c g r n : Nat) (hb : b < batches) (hc : c < nchunks) (hhead : g < p.groups)
    (hr : r < min p.chunk (p.seqlen - c * p.chunk)) (hn : n < p.K) :
    ∃ q ∈ HostGrid.programs (HostGrid.dimensions BM BN batches nchunks p),
      ∃ i : TileIndex [BM, BN], HostGrid.active BM BN p q i ∧
        q.batch = b ∧ RealViews.chunk p q.chunkHead = c ∧ RealViews.head p q.chunkHead = g ∧
        RealViews.row BM BN p q.tile i.1 = r ∧ RealViews.column BN p q.tile i.2.1 = n := by
  let cols := (p.K + BN - 1) / BN
  let rows := (p.chunk + BM - 1) / BM
  have hcol : n / BN < cols := quotient_bound p.K BN n hBN hn
  have hcols : 0 < cols := lt_of_le_of_lt (Nat.zero_le _) hcol
  have hrow : r / BM < rows := quotient_bound p.chunk BM r hBM (lt_of_lt_of_le hr (Nat.min_le_left _ _))
  let q : HostGrid.Pid := ⟨r / BM * cols + n / BN, b, c * p.groups + g⟩
  let i : TileIndex [BM, BN] := (⟨r % BM, Nat.mod_lt _ hBM⟩, ⟨n % BN, Nat.mod_lt _ hBN⟩, PUnit.unit)
  have htile : q.tile < rows * cols := by
    calc
      q.tile < (r / BM + 1) * cols := by dsimp [q]; rw [Nat.add_mul, Nat.one_mul]; omega
      _ ≤ rows * cols := Nat.mul_le_mul_right cols (by omega)
  have hch : q.chunkHead < nchunks * p.groups := by
    calc
      q.chunkHead < (c + 1) * p.groups := by dsimp [q]; rw [Nat.add_mul, Nat.one_mul]; omega
      _ ≤ nchunks * p.groups := Nat.mul_le_mul_right p.groups (by omega)
  have hqd : q.tile / cols = r / BM := by
    dsimp [q]
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hcols, Nat.div_eq_of_lt hcol]
    simp
  have hqm : q.tile % cols = n / BN := by
    dsimp [q]
    rw [Nat.mul_add_mod_self_right, Nat.mod_eq_of_lt hcol]
  have hchunk : RealViews.chunk p q.chunkHead = c := by
    dsimp [RealViews.chunk, q]
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hg, Nat.div_eq_of_lt hhead]
    simp
  have hgroup : RealViews.head p q.chunkHead = g := by
    unfold RealViews.head
    rw [hchunk]
    dsimp [q]
    omega
  have hrows : RealViews.row BM BN p q.tile i.1 = r := by
    change q.tile / cols * BM + r % BM = r
    rw [hqd, Nat.div_add_mod']
  have hcolumns : RealViews.column BN p q.tile i.2.1 = n := by
    change q.tile % cols * BN + n % BN = n
    rw [hqm, Nat.div_add_mod']
  refine ⟨q, (HostGrid.mem_programs _ _).mpr ⟨htile,hb,hch⟩, i, ?_, rfl, hchunk, hgroup, hrows, hcolumns⟩
  change RealViews.row BM BN p q.tile i.1 < RealViews.length p q.chunkHead ∧ _
  simpa [hrows, hcolumns, RealViews.length, RealMath.chunkLimit, hchunk] using And.intro hr hn

/-- Legal host shape reaches every sequence position; no unwritten output is
silently promoted into the active target. -/
theorem chunk_for_sequence (seqlen chunk position : Nat) (positive : 0 < chunk) (inside : position < seqlen) :
    position / chunk < (seqlen + chunk - 1) / chunk ∧
      position % chunk < min chunk (seqlen - (position / chunk) * chunk) ∧
      (position / chunk) * chunk + position % chunk = position := by
  have hm := Nat.mod_lt position positive
  have he := Nat.div_add_mod' position chunk
  exact ⟨quotient_bound seqlen chunk position positive inside, by omega, he⟩

#axiomsClean covers
#axiomsClean chunk_for_sequence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostCoverage
