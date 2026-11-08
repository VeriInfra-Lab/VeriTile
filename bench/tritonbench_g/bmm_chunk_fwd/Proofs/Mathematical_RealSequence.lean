import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealLoop
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealMath

/-! The actual optional signed sequence-ID filter. Inactive row and column
loads use -1 and -2 respectively; they are not converted to natural numbers.
The resulting masked contraction is connected to the independent formula. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSequence
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

private theorem signed_extent (chunk seqlen used : Nat) :
    (min (chunk : Int) ((seqlen : Int)-(used : Int))).toNat=min chunk (seqlen-used) := by
  simp only [min_def]
  split_ifs <;> omega

def limitOp (seqlen chunk : Nat) : Op .int [] :=
  let rest := Op.sub .int .nil (.castNatToInt (.constNat seqlen))
    (.mul .int .nil (.ref .int [] "pid_c") (.castNatToInt (.constNat chunk)))
  .where (.lt .int .nil (.castNatToInt (.constNat chunk)) rest)
    (.castNatToInt (.constNat chunk)) rest

def indexLoad (n stride : Nat) (offset : RegName) (sentinel : Int) : Op .int [n] :=
  .load .int (.ptr (.ptrAdd .scalarL (.ref .ptr [] "seq_idx_ptr")
    (.castIntToNat (.mul .int .scalarR (.ref .int [n] offset) (.castNatToInt (.constNat stride))))))
    (.maskOther (.lt .int .scalarR (.ref .int [n] offset) (.ref .int [] "chunk_size_limit"))
      ((Op.constInt sentinel).broadcast [n]))

def filtered (BM BN : Nat) : Op .real [BM, BN] :=
  .where (.eq .int Broadcast.nil.consL.consR
    (.expandDim ⟨1, by simp⟩ (.ref .int [BM] "seq_idx_m"))
    (.expandDim ⟨0, by simp⟩ (.ref .int [BN] "seq_idx_n")))
    (.ref .real [BM, BN] "acc") ((Op.const 0.0).broadcast [BM, BN])

def code (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .int [] "chunk_size_limit" (limitOp p.seqlen p.chunk),
   .assign .int [BM] "seq_idx_m" (indexLoad BM p.sRow "offs_m" (-1)),
   .assign .int [BN] "seq_idx_n" (indexLoad BN p.sRow "offs_n" (-2)),
   .assign .real [BM, BN] "acc" (filtered BM BN)]

theorem matches_source (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart ((Source.conditional? (RealProgram.parts swapped BM BN BK p).after
      (fun _ => Bool.true)).getD ⟨[], .constBool Bool.false, [], [], []⟩).yes = code BM BN p := by
  cases swapped
  all_goals
    simp [code, limitOp, indexLoad, filtered, RealProgram.realPart, RealProgram.parts,
      RealProgram.branch, RealProgram.source, TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,VeriTile.Bench.Repairs.BmmChunkFwd.originalKernel,
      VeriTile.Bench.Repairs.BmmChunkFwd.dotFirstKernel,
      Source.conditional?, Source.signedRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try norm_num

def indexValues (s : BlockState) (region : RegionName) (base stride length : Nat)
    {n : Nat} (offsets : Fin n → Nat) (sentinel : Int) : Tile .int [n] :=
  Tile.vec fun i => if offsets i < length then s.readMemValue .int region (base + offsets i * stride)
    else sentinel

theorem load_eval (s : BlockState) (region : RegionName) (n base stride length : Nat)
    (offset : RegName) (offsets : Fin n → Nat) (sentinel : Int)
    (hp : s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (region,base)))
    (ho : s.regs .int [n] offset = some (Tile.vec (fun i => (offsets i : Int))))
    (hl : RealLoads.LimitAt s length) :
    evalOp (indexLoad n stride offset sentinel) s =
      some (indexValues s region base stride length offsets sentinel) := by
  obtain ⟨raw,hl,hrange⟩ := hl
  simp [indexLoad, evalOp.eq_def, hp, ho, hl, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [indexValues, Tile.ptrAdd, Tile.bop, Tile.cop, Tile.vec, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, ComparableDType.lt]
  have hm : (offsets i : Int)<raw ↔ offsets i<length := by rw [←hrange];omega
  simp only [hm,← Int.natCast_mul,Int.toNat_natCast]

theorem filtered_eval (s : BlockState) (BM BN : Nat) (m : Fin BM → Int) (n : Fin BN → Int)
    (carry : Fin BM → Fin BN → ℝ)
    (hm : s.regs .int [BM] "seq_idx_m" = some (Tile.vec m))
    (hn : s.regs .int [BN] "seq_idx_n" = some (Tile.vec n))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    evalOp (filtered BM BN) s =
      some (Tile.mat fun i j => some (if m i = n j then carry i j else 0)) := by
  simp [filtered, evalOp.eq_def, hm, hn, ha, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.eq, TileShape.dropInsertedIndex,
    Tile.select]
  by_cases h : m i = n j <;> simp [h] <;> norm_num

theorem limit_eval (s : BlockState) (seqlen chunk pc : Nat)
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int))) :
    evalOp (limitOp seqlen chunk) s = some (Tile.scalar
      (min (chunk : Int) ((seqlen : Int)-(pc : Int)*(chunk : Int)))) := by
  simp [limitOp,evalOp.eq_def,hc,bind,Option.bind,Tile.bop,Tile.cop,Tile.scalar,
    Tile.select,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul,
    NumericDType.sub,ComparableDType.lt,←min_def_lt,-Tile.scalar_eta]

def result (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (pc base : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let length := min p.chunk (p.seqlen - pc * p.chunk)
  let m := indexValues s (Region.cast p.Seq) base p.sRow length rows (-1)
  let n := indexValues s (Region.cast p.Seq) base p.sRow length columns (-2)
  let t := s.setReg "chunk_size_limit" .int []
    (Tile.scalar (min (p.chunk : Int) ((p.seqlen : Int)-(pc : Int)*(p.chunk : Int))))
  let t := t.setReg "seq_idx_m" .int [BM] m
  let t := t.setReg "seq_idx_n" .int [BN] n
  t.setReg "acc" .real [BM, BN] (Tile.mat fun i j => some
    (if m.data (i,PUnit.unit) = n.data (j,PUnit.unit) then carry i j else 0))

theorem executes (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (pc base : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ)
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hp : s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (Region.cast p.Seq,base)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec (fun i => (rows i : Int))))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => (columns i : Int))))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (code BM BN p) s = some (result s BM BN p pc base rows columns carry) := by
  let length := min p.chunk (p.seqlen - pc * p.chunk)
  let m := indexValues s (Region.cast p.Seq) base p.sRow length rows (-1)
  let n := indexValues s (Region.cast p.Seq) base p.sRow length columns (-2)
  let t := s.setReg "chunk_size_limit" .int []
    (Tile.scalar (min (p.chunk : Int) ((p.seqlen : Int)-(pc : Int)*(p.chunk : Int))))
  let u := t.setReg "seq_idx_m" .int [BM] m
  let v := u.setReg "seq_idx_n" .int [BN] n
  have hc' := limit_eval s p.seqlen p.chunk pc hc
  have hm' : evalOp (indexLoad BM p.sRow "offs_m" (-1)) t = some m := by
    apply load_eval
    · simpa [t] using hp
    · simpa [t] using hr
    · refine ⟨min (p.chunk : Int) ((p.seqlen : Int)-(pc : Int)*(p.chunk : Int)), by simp [t], ?_⟩
      simpa only [length,←Int.natCast_mul] using signed_extent p.chunk p.seqlen (pc*p.chunk)
  have hn' : evalOp (indexLoad BN p.sRow "offs_n" (-2)) u = some n := by
    apply load_eval
    · simpa [u, t] using hp
    · simpa [u, t] using hn
    · refine ⟨min (p.chunk : Int) ((p.seqlen : Int)-(pc : Int)*(p.chunk : Int)), by simp [u,t], ?_⟩
      simpa only [length,←Int.natCast_mul] using signed_extent p.chunk p.seqlen (pc*p.chunk)
  have hf : evalOp (filtered BM BN) v = some (Tile.mat fun i j => some
      (if m.data (i,PUnit.unit) = n.data (j,PUnit.unit) then carry i j else 0)) := by
    apply filtered_eval
    · simp [v, u]
      rfl
    · simp [v]
      rfl
    · simpa [v, u, t] using ha
  dsimp only [v, u, t, length] at hm' hn' hf
  simp only [code, stepStmts, stepStmt, hc', hm', hn', hf, bind, Option.bind_some]
  rfl

theorem state_frame (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (pc base : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ) :
    {result s BM BN p pc base rows columns carry with regs := s.regs} = s := by
  cases s
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSequence
