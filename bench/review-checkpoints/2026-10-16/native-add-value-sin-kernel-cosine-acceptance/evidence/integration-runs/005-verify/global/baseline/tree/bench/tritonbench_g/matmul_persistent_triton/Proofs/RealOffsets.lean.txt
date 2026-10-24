import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealGroup
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealStore

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealOffsets
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def code (p : Parameters) : List Stmt := (RealGroup.setup p).drop 8

def result (p : Parameters) (s : BlockState) (row column : Nat) : BlockState :=
  let rs := RealStore.rows p row
  let cs := RealStore.columns p column
  let ar := Tile.vec fun i => (RealTiles.clamp p.M (rs i) : Int)
  let bc := Tile.vec fun j => (RealTiles.clamp p.N (cs j) : Int)
  let t := s.setReg "offs_am" .int [p.BM] (Tile.vec fun i => (rs i : Int))
  let t := t.setReg "offs_bn" .int [p.BN] (Tile.vec fun j => (cs j : Int))
  let t := t.setReg "offs_am" .int [p.BM] ar
  let t := t.setReg "offs_bn" .int [p.BN] bc
  let t := t.setReg "offs_am" .int [p.BM] ar
  t.setReg "offs_bn" .int [p.BN] bc

/-- Actual source row/column clamping, including both retained scheduling-hint
assignments. Only the later K mask zero-fills input loads. -/
theorem executes (p : Parameters) (s : BlockState) (row column : Nat)
    (hr : s.regs .int [] "start_m" = some (Tile.scalar (row*p.BM : Nat)))
    (hc : s.regs .int [] "start_n" = some (Tile.scalar (column*p.BN : Nat))) :
    stepStmts (code p) s = some (result p s row column) := by
  simp only [code, RealGroup.setup, RealProgram.body, RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType,
    List.getElem!_cons_zero, List.getElem!_cons_succ, RealProgram.loopBody, RealProgram.thenBody,
    List.take, List.drop]
  simp [result, stepStmts, stepStmt, evalOp.eq_def, hr, hc, bind, Option.bind,
    Tile.scalar, Tile.vec, Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    RealStore.rows, RealStore.columns, RealTiles.clamp, -Tile.scalar_eta]
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt]
  try rfl

theorem frame (p : Parameters) (s : BlockState) (row column : Nat) :
    {result p s row column with regs := s.regs} = s := rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealOffsets
