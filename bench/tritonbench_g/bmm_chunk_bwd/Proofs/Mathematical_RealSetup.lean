import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealProgram
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Sources.Mathematical_RealMath
import VeriTile.Triton
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.IntegerIndices

/-! Execute the actual symbolic pre-loop source. It derives the tile, batch,
chunk and group indices, pointer grids, runtime chunk limit and zero carry
from the initial state. No pre-populated register or execution assumption is
used. Setup performs no memory access. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSetup
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 250000
set_option linter.unusedSimpArgs false

def chunk (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 / p.groups
def head (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 - chunk s p * p.groups
def tilesN (BN : Nat) (p : RealProgram.Parameters) := (p.K + BN - 1) / BN
def tileM (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 / tilesN BN p
def tileN (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 % tilesN BN p
def rows (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Fin BM → Nat :=
  fun i => tileM s BN p * BM + i.val
def columns (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) : Fin BN → Nat :=
  fun j => tileN s BN p * BN + j.val
def limit (s : BlockState) (p : RealProgram.Parameters) :=
  RealMath.chunkLimit p.seqlen p.chunk (chunk s p)
/-- The original signed subtraction is retained, including out-of-range CTAs. -/
def limitInt (s : BlockState) (p : RealProgram.Parameters) : Int :=
  min (p.chunk : Int) ((p.seqlen : Int) - (chunk s p : Int) * p.chunk)

@[simp] theorem limitInt_toNat (s : BlockState) (p : RealProgram.Parameters) :
    (limitInt s p).toNat = limit s p := by
  change (min (p.chunk : Int) ((p.seqlen : Int) - (chunk s p : Int) * p.chunk)).toNat =
    min p.chunk (p.seqlen - chunk s p * p.chunk)
  rw [← Int.natCast_mul]
  simp only [min_def]
  split_ifs <;> omega

def aBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.aBatch + chunk s p * p.chunk * p.aRow + head s p * p.aHead
def dBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.dBatch + chunk s p * p.dChunk + head s p * p.dHead
def aPointers (s : BlockState) (BN BCS : Nat) (p : RealProgram.Parameters) : Tile .ptr [BCS, BN] :=
  Tile.mat fun k j => (p.A, aBase s p + (k.val * p.aRow + columns s BN p j * p.aColumn))
def dPointers (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM, BCS] :=
  Tile.mat fun i k => (p.Dout, dBase s p + (rows s BM BN p i * p.dColumn + k.val * p.dRow))

def arguments (s : BlockState) (p : RealProgram.Parameters) : BlockState :=
  let t := s.setReg "seqlen" .int [] (Tile.scalar (p.seqlen : Int))
  let t := t.setReg "chunk_size" .int [] (Tile.scalar (p.chunk : Int))
  let t := t.setReg "K" .int [] (Tile.scalar (p.K : Int))
  let t := t.setReg "ngroups" .int [] (Tile.scalar (p.groups : Int))
  let t := t.setReg "stride_a_batch" .int [] (Tile.scalar (p.aBatch : Int))
  let t := t.setReg "stride_a_seqlen" .int [] (Tile.scalar (p.aRow : Int))
  let t := t.setReg "stride_a_head" .int [] (Tile.scalar (p.aHead : Int))
  let t := t.setReg "stride_ak" .int [] (Tile.scalar (p.aColumn : Int))
  let t := t.setReg "stride_dout_batch" .int [] (Tile.scalar (p.dBatch : Int))
  let t := t.setReg "stride_dout_chunk" .int [] (Tile.scalar (p.dChunk : Int))
  let t := t.setReg "stride_dout_head" .int [] (Tile.scalar (p.dHead : Int))
  let t := t.setReg "stride_dout_csize_m" .int [] (Tile.scalar (p.dRow : Int))
  let t := t.setReg "stride_dout_csize_n" .int [] (Tile.scalar (p.dColumn : Int))
  let t := t.setReg "stride_db_batch" .int [] (Tile.scalar (p.oBatch : Int))
  let t := t.setReg "stride_db_seqlen" .int [] (Tile.scalar (p.oRow : Int))
  let t := t.setReg "stride_db_head" .int [] (Tile.scalar (p.oHead : Int))
  let t := t.setReg "stride_db_k" .int [] (Tile.scalar (p.oColumn : Int))
  let t := t.setReg "stride_res_batch" .int [] (Tile.scalar (p.rBatch : Int))
  let t := t.setReg "stride_res_seqlen" .int [] (Tile.scalar (p.rRow : Int))
  let t := t.setReg "stride_res_head" .int [] (Tile.scalar (p.rHead : Int))
  let t := t.setReg "stride_res_k" .int [] (Tile.scalar (p.rColumn : Int))
  t

def result (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := (arguments s p).setReg "pid_b" .int [] (Tile.scalar (s.pids 1 : Int))
  let t := t.setReg "pid_ch" .int [] (Tile.scalar (s.pids 2 : Int))
  let t := t.setReg "pid_c" .int [] (Tile.scalar (chunk s p : Int))
  let t := t.setReg "pid_h" .int [] (Tile.scalar (head s p : Int))
  let t := t.setReg "num_pid_n" .int [] (Tile.scalar (tilesN BN p : Int))
  let t := t.setReg "pid_m" .int [] (Tile.scalar (tileM s BN p : Int))
  let t := t.setReg "pid_n" .int [] (Tile.scalar (tileN s BN p : Int))
  let t := t.setReg "a_ptr" .ptr [] (Tile.scalar (p.A, aBase s p))
  let t := t.setReg "dout_ptr" .ptr [] (Tile.scalar (p.Dout, dBase s p))
  let t := t.setReg "offs_m" .int [BM] (Tile.vec fun i => (rows s BM BN p i : Int))
  let t := t.setReg "offs_n" .int [BN] (Tile.vec fun j => (columns s BN p j : Int))
  let t := t.setReg "offs_cs" .int [BCS] (Tile.vec fun k => (k.val : Int))
  let t := t.setReg "dout_ptrs" .ptr [BM, BCS] (dPointers s BM BN BCS p)
  let t := t.setReg "a_ptrs" .ptr [BCS, BN] (aPointers s BN BCS p)
  let t := t.setReg "chunk_size_limit" .int [] (Tile.scalar (limitInt s p))
  t.setReg "acc" .real [BM, BN] (Tile.mat fun _ _ => some 0)

def code (BM BN BCS : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealProgram.realPart (RealProgram.parts Bool.false BM BN BCS p).before

theorem original_candidate_setup (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).before = code BM BN BCS p := by
  cases swapped <;> rfl

theorem executes (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    stepStmts (code BM BN BCS p) s = some (result s BM BN BCS p) := by
  simp [code, RealProgram.realPart, RealProgram.parts, RealProgram.source,
    TypedSource.specialize, VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.family,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.sourceKernel,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.dotDType, Source.intRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
  simp [result, arguments, stepStmts, stepStmt, eval_truncDiv, eval_truncMod, evalOp.eq_def, bind, Option.bind,
    chunk, head, tilesN, tileM, tileN, rows, columns, aBase, dBase, aPointers, dPointers]
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Tile.mat,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
    NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
    Nat.zero_add, Nat.add_assoc, Int.tdiv, Int.tmod]
  have cast_not_neg (n : Nat) : ¬ (n : Int) < 0 := Int.not_lt.mpr (Int.natCast_nonneg n)
  have hdiv : (p.K + (BN - 1)) / BN = (p.K + BN - 1) / BN := by
    by_cases h : BN = 0
    · simp [h]
    · congr 1; omega
  simp only [Tile.cop, Tile.select, ComparableDType.lt, IntegerExpr.same_left,
    IntegerExpr.same_right, add_zero, zero_add, decide_eq_true_eq,
    ← Int.natCast_add, cast_not_neg, if_false]
  simp only [← Int.natCast_ediv, ← Int.natCast_emod, IntegerIndices.head_cast, IntegerIndices.remainder_cast,
    ← Int.natCast_add, ← Int.natCast_mul, Int.toNat_natCast, hdiv]
  simp only [IntegerIndices.source_min, limitInt, chunk, Nat.add_assoc]
  simp only [IntegerIndices.head_sub_cast, IntegerIndices.remainder_sub_cast,
    ← Int.natCast_add, ← Int.natCast_mul, Int.toNat_natCast, Nat.add_assoc]

theorem state_frame (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    {result s BM BN BCS p with regs := s.regs} = s := by
  cases s
  rfl

theorem chunk_limit (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    (result s BM BN BCS p).regs .int [] "chunk_size_limit" = some (Tile.scalar (limitInt s p)) := by
  simp [result]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSetup
