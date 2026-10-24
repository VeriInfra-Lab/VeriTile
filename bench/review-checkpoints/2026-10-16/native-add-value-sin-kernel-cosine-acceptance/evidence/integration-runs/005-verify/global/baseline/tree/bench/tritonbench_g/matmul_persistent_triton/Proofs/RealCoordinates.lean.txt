import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealGroup

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealCoordinates
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def code (p : Parameters) : List Stmt := ((RealGroup.setup p).drop 4).take 4

def result (p : Parameters) (s : BlockState) (tile : Nat) : BlockState :=
  let nr := RealGroup.rows p
  let nc := RealGroup.columns p
  let r := Schedule.row nr nc p.GM tile
  let c := Schedule.column nr nc p.GM tile
  let t := s.setReg "pid_m" .int [] (Tile.scalar (r : Int))
  let t := t.setReg "pid_n" .int [] (Tile.scalar (c : Int))
  let t := t.setReg "start_m" .int [] (Tile.scalar (r*p.BM : Nat))
  t.setReg "start_n" .int [] (Tile.scalar (c*p.BN : Nat))

theorem signed_remainder (x y : Nat) :
    (x : Int) - (y : Int)*((x : Int)/(y : Int)) = (x%y : Nat) := by
  rw [Int.natCast_emod, Int.emod_def]

/-- The actual signed pid_m/pid_n arithmetic implements the independent grouped
coordinates, preserving the absolute-tile remainder in the original source. -/
theorem executes (p : Parameters) (s : BlockState) (tile : Nat)
    (ht : s.regs .int [] "tile_id" = some (Tile.scalar (tile : Int)))
    (hb : s.regs .int [] "first_pid_m" = some (Tile.scalar
      (Schedule.groupBase (RealGroup.rows p) (RealGroup.columns p) p.GM tile : Int)))
    (hs : s.regs .int [] "group_size_m" = some (Tile.scalar
      (Schedule.groupSize (RealGroup.rows p) (RealGroup.columns p) p.GM tile : Int)))
    (hg : s.regs .nat [] "num_pid_in_group" = some (Tile.scalar (p.GM*RealGroup.columns p))) :
    stepStmts (code p) s = some (result p s tile) := by
  have hnonnegative : ¬ (tile : Int) < 0 := by omega
  have hrem (y : Nat) : ¬ (tile : Int) < (y : Int)*((tile : Int)/(y : Int)) := by
    have h : y * (tile / y) ≤ tile := by
      simpa [Nat.mul_comm] using Nat.div_mul_le_self tile y
    exact_mod_cast (not_lt_of_ge h)
  have hgroup := hrem (p.GM*RealGroup.columns p)
  simp only [Int.natCast_mul] at hgroup
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
  simp [result, stepStmts, stepStmt, evalOp.eq_def, ht, hb, hs, hg, bind, Option.bind,
    Tile.scalar, Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.sub, NumericDType.mul, IntegralDType.floorDiv,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, IntegerExpr.sameShape,
    Schedule.row, Schedule.column, hnonnegative, hgroup, ← Int.emod_def, -Tile.scalar_eta]
  try rfl

theorem frame (p : Parameters) (s : BlockState) (tile : Nat) :
    {result p s tile with regs := s.regs} = s := rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealCoordinates
