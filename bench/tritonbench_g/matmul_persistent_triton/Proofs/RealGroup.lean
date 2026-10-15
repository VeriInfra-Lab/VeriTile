import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealProgram
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealMath
import bench.tritonbench_g.matmul_persistent_triton.Proofs.Schedule

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealGroup
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
local instance : Inhabited Stmt := ⟨.assign .nat [] "" (.constNat 0)⟩

def rows (p : Parameters) : Nat := RealMath.blocks p.M p.BM
def columns (p : Parameters) : Nat := RealMath.blocks p.N p.BN

def setup (p : Parameters) : List Stmt := RealProgram.thenBody (RealProgram.body p)[1]!
def code (p : Parameters) : List Stmt := (setup p).take 4

def result (p : Parameters) (s : BlockState) (tile : Nat) : BlockState :=
  let nr := RealGroup.rows p
  let nc := RealGroup.columns p
  let t := s.setReg "tile_id" .int [] (Tile.scalar (tile : Int))
  let t := t.setReg "group_id" .int [] (Tile.scalar (tile / (p.GM*nc) : Nat))
  let t := t.setReg "first_pid_m" .int [] (Tile.scalar (Schedule.groupBase nr nc p.GM tile : Int))
  t.setReg "group_size_m" .int [] (Tile.scalar (Schedule.groupSize nr nc p.GM tile : Int))

/-- Actual source group setup, including the shortened final row group.
The incoming signed tile can still be negative on the first output tile. -/
theorem executes (p : Parameters) (s : BlockState) (tile : Nat)
    (hGM : 0 < p.GM) (htile : tile < RealGroup.rows p * RealGroup.columns p)
    (ht : s.regs .int [] "tile_id" = some (Tile.scalar ((tile : Int)-(p.SMS : Int))))
    (hr : s.regs .nat [] "num_pid_m" = some (Tile.scalar (RealGroup.rows p)))
    (hg : s.regs .nat [] "num_pid_in_group" = some (Tile.scalar (p.GM*RealGroup.columns p))) :
    stepStmts (code p) s = some (result p s tile) := by
  have hnonnegative : ¬ (tile : Int) < 0 := by omega
  have hbounds := Schedule.grouped_bounds (RealGroup.rows p) (RealGroup.columns p) p.GM tile hGM htile
  have hb : Schedule.groupBase (RealGroup.rows p) (RealGroup.columns p) p.GM tile ≤ RealGroup.rows p := by
    have hh := hbounds.2.1
    unfold Schedule.row at hh
    omega
  have hcompare : ((RealGroup.rows p : Int) - (Schedule.groupBase (RealGroup.rows p) (RealGroup.columns p) p.GM tile : Int) < (p.GM : Int)) ↔
      RealGroup.rows p - Schedule.groupBase (RealGroup.rows p) (RealGroup.columns p) p.GM tile < p.GM := by omega
  simp only [Schedule.groupBase] at hcompare
  have hsub := Int.natCast_sub hb
  simp only [Schedule.groupBase] at hsub
  simp only [Int.natCast_mul, Int.natCast_ediv] at hcompare hsub
  simp only [code, setup, RealProgram.body, RealSource.kernel,
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
  simp [result, stepStmts, stepStmt, evalOp.eq_def, ht, hr, hg, bind, Option.bind,
    Tile.scalar, Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.sub, NumericDType.mul, IntegralDType.floorDiv,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, IntegerExpr.sameShape,
    Schedule.groupBase, Schedule.groupSize, min_def_lt,
    hnonnegative, hsub, hcompare, -Tile.scalar_eta]
  try rfl

theorem frame (p : Parameters) (s : BlockState) (tile : Nat) :
    {result p s tile with regs := s.regs} = s := rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealGroup
