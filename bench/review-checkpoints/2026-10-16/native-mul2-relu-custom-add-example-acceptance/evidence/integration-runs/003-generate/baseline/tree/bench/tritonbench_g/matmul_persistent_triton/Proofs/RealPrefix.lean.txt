import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealSource
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealMath
import VeriTile.Triton.Semantics.Step

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealPrefix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def rows (p : Parameters) : Nat := RealMath.blocks p.M p.BM
def columns (p : Parameters) : Nat := RealMath.blocks p.N p.BN
def kTiles (p : Parameters) : Nat := RealMath.blocks p.K p.BK
def total (p : Parameters) : Nat := rows p * columns p

def result (p : Parameters) (s : BlockState) : BlockState :=
  let t := s.setReg "start_pid" .nat [] (Tile.scalar (s.pids 0))
  let t := t.setReg "num_pid_m" .nat [] (Tile.scalar (rows p))
  let t := t.setReg "num_pid_n" .nat [] (Tile.scalar (columns p))
  let t := t.setReg "k_tiles" .nat [] (Tile.scalar (kTiles p))
  let t := t.setReg "num_tiles" .nat [] (Tile.scalar (total p))
  let t := t.setReg "tiles_per_SM" .nat [] (Tile.scalar (total p / p.SMS))
  let t := if s.pids 0 < total p % p.SMS then
    t.setReg "tiles_per_SM" .nat [] (Tile.scalar (total p / p.SMS + 1)) else t
  let t := t.setReg "tile_id" .int [] (Tile.scalar ((s.pids 0 : Int) - p.SMS))
  let t := t.setReg "ki" .int [] (Tile.scalar (-1))
  let t := t.setReg "offs_k_for_mask" .nat [p.BK] (Tile.vec Fin.val)
  let t := t.setReg "num_pid_in_group" .nat [] (Tile.scalar (p.GM * columns p))
  let t := t.setReg "pid_m" .int [] (Tile.scalar 0)
  let t := t.setReg "pid_n" .int [] (Tile.scalar 0)
  let t := t.setReg "offs_am" .int [p.BM] (Tile.vec fun i => (i.val : Int))
  let t := t.setReg "offs_bn" .int [p.BN] (Tile.vec fun i => (i.val : Int))
  t.setReg "accumulator" .real [p.BM, p.BN] (Tile.mat fun _ _ => some 0)

def code (p : Parameters) : List Stmt := (RealSource.kernel p Bool.false).body.take 16

/-- The actual complete source prefix executes from every initial real state;
all dimensions, strides and configuration remain symbolic. -/
theorem executes (p : Parameters) (s : BlockState) :
    stepStmts (code p) s = some (result p s) := by
  simp only [code, RealSource.kernel, Parameters.source, TypedSource.kernel,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    List.map_cons, List.map_nil, FloatDType.toTileDType]
  by_cases h : s.pids 0 < total p % p.SMS
  all_goals simp only [result, rows, columns, kTiles, total, RealMath.blocks] at h ⊢
  all_goals simp [stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
    Tile.ptrAdd, Tile.bop, Tile.cop, Tile.scalar, Tile.vec, Tile.mat,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub,
    NumericDType.mul, NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
    ComparableDType.lt, ComparableDType.eq, -Tile.scalar_eta, h]
  all_goals try rfl

/-- Prefix evaluation changes only registers, preserving the complete source
memory and all host program coordinates. -/
theorem frame (p : Parameters) (s : BlockState) :
    {result p s with regs := s.regs} = s := by
  unfold result
  split <;> rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealPrefix
