import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealSetup

/-! Values of the two actual contraction loads. The residual tail predicate
is retained even in the last partial block; row and column masking are
independent. Values come from the current memory, never a supplied dot result. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoads
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def remaining (BCS : Nat) : Op .int [] :=
  .sub .int .nil (.ref .int [] "chunk_size_limit")
    (.mul .int .nil (.ref .int [] "cs") (.castNatToInt (.constNat BCS)))

def doutMask (BM BCS chunk : Nat) : Op .bool [BM, BCS] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .int .scalarR (.expandDim ⟨1, by simp⟩ (.ref .int [BM] "offs_m")) (.castNatToInt (.constNat chunk)))
    (.lt .int .scalarR (.expandDim ⟨0, by simp⟩ (.ref .int [BCS] "offs_cs")) (remaining BCS))

def aMask (BN BCS K : Nat) : Op .bool [BCS, BN] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .int .scalarR (.expandDim ⟨1, by simp⟩ (.ref .int [BCS] "offs_cs")) (remaining BCS))
    (.lt .int .scalarR (.expandDim ⟨0, by simp⟩ (.ref .int [BN] "offs_n")) (.castNatToInt (.constNat K)))

private theorem nat_lt_signed_sub (x c b : Nat) (length : Int) :
    (x : Int) < length - (c : Int) * b ↔ x < length.toNat - c*b := by
  rw [← Int.natCast_mul]
  omega

def doutLoad (BM BCS chunk : Nat) : Op .real [BM, BCS] :=
  .load .real (.ptr (.ref .ptr [BM, BCS] "dout_ptrs"))
    (.maskOther (doutMask BM BCS chunk) ((Op.const 0).broadcast [BM, BCS]))

def aLoad (BN BCS K : Nat) : Op .real [BCS, BN] :=
  .load .real (.ptr (.ref .ptr [BCS, BN] "a_ptrs"))
    (.maskOther (aMask BN BCS K) ((Op.const 0).broadcast [BCS, BN]))

theorem dout_mask (s : BlockState) (BM BCS chunk : Nat) (length : Int) (c : Nat) (rows : Fin BM → Nat)
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rows i : Int)))
    (hk : s.regs .int [BCS] "offs_cs" = some (Tile.vec fun k => (k.val : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .int [] "cs" = some (Tile.scalar (c : Int))) :
    evalOp (doutMask BM BCS chunk) s = some (Tile.mat fun i k =>
      decide (rows i < chunk ∧ k.val < length.toNat - c * BCS)) := by
  simp [doutMask, remaining, evalOp.eq_def, hr, hk, hl, hc, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i, k, ⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, NumericDType.sub, NumericDType.mul, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex]
  simp only [Int.ofNat_lt, nat_lt_signed_sub]

theorem a_mask (s : BlockState) (BN BCS K : Nat) (length : Int) (c : Nat) (columns : Fin BN → Nat)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun j => (columns j : Int)))
    (hk : s.regs .int [BCS] "offs_cs" = some (Tile.vec fun k => (k.val : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .int [] "cs" = some (Tile.scalar (c : Int))) :
    evalOp (aMask BN BCS K) s = some (Tile.mat fun k j =>
      decide (k.val < length.toNat - c * BCS ∧ columns j < K)) := by
  simp [aMask, remaining, evalOp.eq_def, hn, hk, hl, hc, bind, Option.bind]
  apply Tile.ext
  rintro ⟨k, j, ⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, NumericDType.sub, NumericDType.mul, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex]
  simp only [Int.ofNat_lt, nat_lt_signed_sub]

def doutValues {BM BCS : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BCS])
    (rows : Fin BM → Nat) (chunk length c : Nat) : Tile .real [BM, BCS] :=
  Tile.mat fun i k => some (if rows i < chunk ∧ k.val < length - c * BCS then
    s.readMem (ptrs.data (i,k,PUnit.unit)).1 (ptrs.data (i,k,PUnit.unit)).2 else 0)

def aValues {BN BCS : Nat} (s : BlockState) (ptrs : Tile .ptr [BCS, BN])
    (columns : Fin BN → Nat) (K length c : Nat) : Tile .real [BCS, BN] :=
  Tile.mat fun k j => some (if k.val < length - c * BCS ∧ columns j < K then
    s.readMem (ptrs.data (k,j,PUnit.unit)).1 (ptrs.data (k,j,PUnit.unit)).2 else 0)

theorem dout_load (s : BlockState) (BM BCS chunk : Nat) (length : Int) (c : Nat) (rows : Fin BM → Nat)
    (ptrs : Tile .ptr [BM, BCS])
    (hp : s.regs .ptr [BM, BCS] "dout_ptrs" = some ptrs)
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rows i : Int)))
    (hk : s.regs .int [BCS] "offs_cs" = some (Tile.vec fun k => (k.val : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .int [] "cs" = some (Tile.scalar (c : Int))) :
    evalOp (doutLoad BM BCS chunk) s = some (doutValues s ptrs rows chunk length.toNat c) := by
  have hm := dout_mask s BM BCS chunk length c rows hr hk hl hc
  have hz : evalOp ((Op.const 0).broadcast [BM, BCS]) s =
      some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BM, BCS]) := by
    simp [evalOp, evalOp_const, Option.bind]
  unfold doutLoad
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hm, hz, bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i, k, ⟨⟩⟩
  by_cases h : rows i < chunk ∧ k.val < length.toNat - c * BCS <;>
    simp [doutValues, Tile.mat, h]

theorem a_load (s : BlockState) (BN BCS K : Nat) (length : Int) (c : Nat) (columns : Fin BN → Nat)
    (ptrs : Tile .ptr [BCS, BN])
    (hp : s.regs .ptr [BCS, BN] "a_ptrs" = some ptrs)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun j => (columns j : Int)))
    (hk : s.regs .int [BCS] "offs_cs" = some (Tile.vec fun k => (k.val : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .int [] "cs" = some (Tile.scalar (c : Int))) :
    evalOp (aLoad BN BCS K) s = some (aValues s ptrs columns K length.toNat c) := by
  have hm := a_mask s BN BCS K length c columns hn hk hl hc
  have hz : evalOp ((Op.const 0).broadcast [BCS, BN]) s =
      some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BCS, BN]) := by
    simp [evalOp, evalOp_const, Option.bind]
  unfold aLoad
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hm, hz, bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k, j, ⟨⟩⟩
  by_cases h : k.val < length.toNat - c * BCS ∧ columns j < K <;>
    simp [aValues, Tile.mat, h]

def code (BM BN BCS : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .real [BM, BCS] "__veritile_load_0" (doutLoad BM BCS p.chunk),
   .assign .real [BM, BCS] "dout" (.ref .real [BM, BCS] "__veritile_load_0"),
   .assign .real [BCS, BN] "__veritile_load_1" (aLoad BN BCS p.K),
   .assign .real [BCS, BN] "a" (.ref .real [BCS, BN] "__veritile_load_1"),
   .assign .real [BM, BN] "acc"
     (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "acc")
       (.dot (batch := []) (.ref .real [BM, BCS] "dout") (.ref .real [BCS, BN] "a"))),
   .assign .ptr [BM, BCS] "dout_ptrs"
     (.ptrAdd .scalarR (.ref .ptr [BM, BCS] "dout_ptrs")
       (.mul .nat .nil (.constNat BCS) (.constNat p.dRow))),
   .assign .ptr [BCS, BN] "a_ptrs"
     (.ptrAdd .scalarR (.ref .ptr [BCS, BN] "a_ptrs")
       (.mul .nat .nil (.constNat BCS) (.constNat p.aRow)))]

set_option maxRecDepth 18000 in
set_option maxHeartbeats 3000000 in
/-- The load views belong to the entire actual five-statement loop body,
including the dot update and both pointer advances. -/
theorem matches_source (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).body = code BM BN BCS p := by
  cases swapped
  all_goals
    simp [code, doutLoad, aLoad, doutMask, aMask, remaining,
      RealProgram.realPart, RealProgram.parts, RealProgram.source,
      TypedSource.specialize, VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.family,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.sourceKernel,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.dotDType, Source.intRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoads
