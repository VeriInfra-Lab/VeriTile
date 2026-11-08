import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealProgram
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Sources.Mathematical_RealMath
import VeriTile.Triton

/-! Execute the real grid/causal prologue and the complete active-tile setup.
The runtime chunk limit and zero accumulator are derived from initial state.
No prepared-register or execution premise is required. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSetup
open VeriTile.Triton
open VeriTile.Bench.Repairs.BmmChunkFwd.ControlArithmetic
attribute [local simp 2000] VeriTile.Triton.eval_truncDiv VeriTile.Triton.eval_truncMod
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

private theorem minimum_int (a b : Int) : (if a<b then a else b)=min a b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Int.le_of_lt h)]
  · rw [if_neg h,min_eq_right (by omega)]

def chunk (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 / p.groups
def head (s : BlockState) (p : RealProgram.Parameters) := s.pids 2 - chunk s p * p.groups
def tilesN (BN : Nat) (p : RealProgram.Parameters) := (p.chunk + BN - 1) / BN
def tileM (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 / tilesN BN p
def tileN (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) := s.pids 0 % tilesN BN p
def rows (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Fin BM → Nat :=
  fun i => tileM s BN p * BM + i.val
def columns (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) : Fin BN → Nat :=
  fun j => tileN s BN p * BN + j.val
def active (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Bool :=
  !p.causal || decide (tileN s BN p * BN < (tileM s BN p + 1) * BM)
def limit (s : BlockState) (p : RealProgram.Parameters) :=
  min p.chunk (p.seqlen - chunk s p * p.chunk)
def signedLimit (s : BlockState) (p : RealProgram.Parameters) : Int :=
  min (p.chunk : Int) ((p.seqlen : Int)-(chunk s p : Int)*p.chunk)

def aBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.aBatch + chunk s p * p.chunk * p.aRow + head s p * p.aHead
def bBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.bBatch + chunk s p * p.chunk * p.bRow + head s p * p.bHead
def seqBase (s : BlockState) (p : RealProgram.Parameters) :=
  s.pids 1 * p.sBatch + chunk s p * p.chunk * p.sRow

def prologueResult (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := s.setReg "seqlen" .int [] (Tile.scalar (p.seqlen : Int))
  let t := t.setReg "chunk_size" .int [] (Tile.scalar (p.chunk : Int))
  let t := t.setReg "K" .int [] (Tile.scalar (p.K : Int))
  let t := t.setReg "ngroups" .int [] (Tile.scalar (p.groups : Int))
  let t := t.setReg "stride_a_batch" .int [] (Tile.scalar (p.aBatch : Int))
  let t := t.setReg "stride_a_seqlen" .int [] (Tile.scalar (p.aRow : Int))
  let t := t.setReg "stride_a_head" .int [] (Tile.scalar (p.aHead : Int))
  let t := t.setReg "stride_ak" .int [] (Tile.scalar (p.aColumn : Int))
  let t := t.setReg "stride_b_batch" .int [] (Tile.scalar (p.bBatch : Int))
  let t := t.setReg "stride_b_seqlen" .int [] (Tile.scalar (p.bRow : Int))
  let t := t.setReg "stride_b_head" .int [] (Tile.scalar (p.bHead : Int))
  let t := t.setReg "stride_bk" .int [] (Tile.scalar (p.bColumn : Int))
  let t := t.setReg "stride_out_batch" .int [] (Tile.scalar (p.oBatch : Int))
  let t := t.setReg "stride_out_chunk" .int [] (Tile.scalar (p.oChunk : Int))
  let t := t.setReg "stride_out_head" .int [] (Tile.scalar (p.oHead : Int))
  let t := t.setReg "stride_outm" .int [] (Tile.scalar (p.oRow : Int))
  let t := t.setReg "stride_outn" .int [] (Tile.scalar (p.oColumn : Int))
  let t := t.setReg "stride_seq_idx_batch" .int [] (Tile.scalar (p.sBatch : Int))
  let t := t.setReg "stride_seq_idx_seqlen" .int [] (Tile.scalar (p.sRow : Int))
  let t := t.setReg "pid_b" .int [] (Tile.scalar (s.pids 1 : Int))
  let t := t.setReg "pid_ch" .int [] (Tile.scalar (s.pids 2 : Int))
  let t := t.setReg "pid_c" .int [] (Tile.scalar (chunk s p : Int))
  let t := t.setReg "pid_h" .int [] (Tile.scalar (head s p : Int))
  let t := t.setReg "num_pid_n" .int [] (Tile.scalar (tilesN BN p : Int))
  let t := t.setReg "pid_m" .int [] (Tile.scalar (tileM s BN p : Int))
  let t := t.setReg "pid_n" .int [] (Tile.scalar (tileN s BN p : Int))
  t

def prologue (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealProgram.realPart (RealProgram.branch Bool.false BM BN BK p).before

theorem prologue_executes (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    stepStmts (prologue BM BN BK p) s = some (prologueResult s BM BN p) := by
  simp [prologue, RealProgram.realPart, RealProgram.branch, RealProgram.source,
    TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,VeriTile.Bench.Repairs.BmmChunkFwd.originalKernel, Source.conditional?, Source.signedRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
  cases hca : p.causal <;>
    simp [prologueResult, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
      chunk, head, tilesN, tileM, tileN, active, hca, Tile.bop, Tile.cop, Tile.scalar,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
      NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
      ComparableDType.lt, -Tile.scalar_eta]
  all_goals
    simp only [Int.natCast_sub (Nat.div_mul_le_self (s.pids 2) p.groups),
      Int.tmod_eq_emod_of_nonneg (Int.natCast_nonneg (s.pids 0)),Int.natCast_mul,Int.natCast_div]

def aPointers (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM, BK] :=
  Tile.mat fun i k => (p.A, aBase s p + (rows s BM BN p i * p.aRow + k.val * p.aColumn))
def bPointers (s : BlockState) (BN BK : Nat) (p : RealProgram.Parameters) : Tile .ptr [BK, BN] :=
  Tile.mat fun k j => (p.B, bBase s p + (k.val * p.bColumn + columns s BN p j * p.bRow))

def result (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) : BlockState :=
  let t := prologueResult s BM BN p
  let t := t.setReg "a_ptr" .ptr [] (Tile.scalar (p.A, aBase s p))
  let t := t.setReg "b_ptr" .ptr [] (Tile.scalar (p.B, bBase s p))
  let t := if p.hasSeq then t.setReg "seq_idx_ptr" .ptr [] (Tile.scalar (Region.cast p.Seq, seqBase s p)) else t
  let t := t.setReg "offs_m" .int [BM] (Tile.vec (fun i => (rows s BM BN p i : Int)))
  let t := t.setReg "offs_n" .int [BN] (Tile.vec (fun i => (columns s BN p i : Int)))
  let t := t.setReg "offs_k" .int [BK] (Tile.vec (fun i => (i.val : Int)))
  let t := t.setReg "a_ptrs" .ptr [BM, BK] (aPointers s BM BN BK p)
  let t := t.setReg "b_ptrs" .ptr [BK, BN] (bPointers s BN BK p)
  let t := t.setReg "chunk_size_limit" .int [] (Tile.scalar (signedLimit s p))
  t.setReg "acc" .real [BM, BN] (Tile.mat fun _ _ => some 0)

def code (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealProgram.realPart (RealProgram.parts Bool.false BM BN BK p).before

theorem original_candidate_setup (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before = code BM BN BK p := by
  cases swapped <;> rfl

theorem executes (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    stepStmts (code BM BN BK p) (prologueResult s BM BN p) = some (result s BM BN BK p) := by
  simp [code, RealProgram.realPart, RealProgram.parts, RealProgram.branch, RealProgram.source,
    TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,VeriTile.Bench.Repairs.BmmChunkFwd.originalKernel, Source.conditional?, Source.signedRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
  cases hs : p.hasSeq <;> cases hc : p.causal
  all_goals
    simp [result, prologueResult, hs, hc, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
      chunk, head, tilesN, tileM, tileN, rows, columns, aBase, bBase, seqBase, aPointers, bPointers]
    simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Tile.mat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
      NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
      Nat.zero_add, Nat.add_assoc]
    simp only [Int.natCast_sub (Nat.div_mul_le_self (s.pids 2) p.groups),
      Int.tmod_eq_emod_of_nonneg (Int.natCast_nonneg (s.pids 0)),Int.natCast_mul,Int.natCast_div]
    simp only [signedLimit, chunk,ComparableDType.lt,Tile.select,minimum_int]
    have hhead : (s.pids 2 : Int)-(s.pids 2 : Int)/(p.groups : Int)*(p.groups : Int)=
        (head s p : Int) := by
      simp only [head,chunk,Int.natCast_sub (Nat.div_mul_le_self (s.pids 2) p.groups),
        Int.natCast_mul,Int.natCast_div]
    simp only [hhead,Tile.cop,ComparableDType.lt,Broadcast.leftIndex,
      Broadcast.rightIndex,decide_eq_true_eq,minimum_int]
    simp only [← Int.natCast_add,← Int.natCast_mul,← Int.natCast_div,
      ← Int.natCast_mod,Int.toNat_natCast,Nat.add_assoc]
    all_goals simp only [head,chunk]

theorem state_frame (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    {result s BM BN BK p with regs := s.regs} = s := by
  cases hs : p.hasSeq <;> cases hc : p.causal <;>
    simp [result, prologueResult, hs, hc, BlockState.setReg]

theorem chunk_limit (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    (result s BM BN BK p).regs .int [] "chunk_size_limit" = some (Tile.scalar (signedLimit s p)) := by
  simp [result]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSetup
