import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealSequence
import VeriTile.Triton.Memory.TraceProofs

/-! Signed metadata loads access only the row/column indices selected by the
actual shortened-length masks. The branch's in-body minimum is executed before
checking the loads; no successful metadata load is an assumption. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSequenceSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem index_safe (bounds : RegionBounds) (s : BlockState) (region : RegionName)
    (n base stride length : Nat) (offset : RegName) (offsets : Fin n → Nat) (sentinel : Int)
    (hp : s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (region,base)))
    (ho : s.regs .int [n] offset = some (Tile.vec (fun i => (offsets i : Int))))
    (hl : RealLoads.LimitAt s length)
    (hb : ∀ i, offsets i < length → base + offsets i * stride < bounds region) :
    (RealSequence.indexLoad n stride offset sentinel).SafeAt bounds s := by
  obtain ⟨raw,hl,hrange⟩ := hl
  have hpv : evalOp (Op.ptrAdd .scalarL (.ref .ptr [] "seq_idx_ptr")
      (.castIntToNat (.mul .int .scalarR (.ref .int [n] offset) (.castNatToInt (.constNat stride))))) s =
      some (Tile.vec fun i => (region,base + offsets i * stride)) := by
    simp [evalOp.eq_def, hp, ho, bind, Option.bind]
    apply Tile.ext
    rintro ⟨i,⟨⟩⟩
    simp only [Tile.ptrAdd,Tile.bop,Tile.vec,Tile.scalar,NumericDType.mul,
      Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast]
  have hmv : evalOp (Op.lt .int .scalarR (.ref .int [n] offset) (.ref .int [] "chunk_size_limit")) s =
      some (Tile.vec fun i => decide (offsets i < length)) := by
    simp [evalOp.eq_def, ho, hl, bind, Option.bind]
    apply Tile.ext
    rintro ⟨i,⟨⟩⟩
    simp only [Tile.cop,Tile.vec,Tile.scalar,ComparableDType.lt,
      Broadcast.leftIndex,Broadcast.rightIndex]
    have hm : (offsets i : Int)<raw ↔ offsets i<length := by rw [←hrange];omega
    simp only [hm]
  rw [RealSequence.indexLoad, Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def], ?_⟩
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ptr he i hi
  rw [hpv] at he
  cases he
  obtain ⟨ms, he, hi⟩ := hi
  rw [hmv] at he
  cases he
  exact hb i.1 (by simpa only [Tile.vec, decide_eq_true_eq] using hi)

theorem safe (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (pc base : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (bounds : RegionBounds)
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hp : s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (Region.cast p.Seq,base)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec (fun i => (rows i : Int))))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => (columns i : Int))))
    (hrow : ∀ i, rows i < min p.chunk (p.seqlen - pc * p.chunk) → base + rows i * p.sRow < bounds (Region.cast p.Seq))
    (hcol : ∀ j, columns j < min p.chunk (p.seqlen - pc * p.chunk) → base + columns j * p.sRow < bounds (Region.cast p.Seq)) :
    Stmt.TraceSafeList bounds (RealSequence.code BM BN p) s := by
  have hlimit := RealSequence.limit_eval s p.seqlen p.chunk pc hc
  rw [RealSequence.code, TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hlimit)]
  refine ⟨by simp [Stmt.TraceSafe, RealSequence.limitOp, Op.SafeAt.eq_def], ?_⟩
  apply TraceProofs.assign_cons
  · apply index_safe bounds _ (Region.cast p.Seq) BM base p.sRow (min p.chunk (p.seqlen - pc * p.chunk)) "offs_m" rows (-1)
    · simpa using hp
    · simpa using hr
    · refine ⟨min (p.chunk : Int) ((p.seqlen : Int)-(pc : Int)*p.chunk),by simp,?_⟩
      simp only [min_def,←Int.natCast_mul]
      split_ifs <;> omega
    · exact hrow
  · intro _
    apply TraceProofs.assign_cons
    · apply index_safe bounds _ (Region.cast p.Seq) BN base p.sRow (min p.chunk (p.seqlen - pc * p.chunk)) "offs_n" columns (-2)
      · simpa using hp
      · simpa using hn
      · refine ⟨min (p.chunk : Int) ((p.seqlen : Int)-(pc : Int)*p.chunk),by simp,?_⟩
        simp only [min_def,←Int.natCast_mul]
        split_ifs <;> omega
      · exact hcol
    · intro _
      apply TraceProofs.all_safe
      simp [RealSequence.filtered, Stmt.TraceSafe, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSequenceSafety
