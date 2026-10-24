import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealCoordinates
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealOffsets
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealControl

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSchedule
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def row (p : Parameters) (tile : Nat) :=
  Schedule.row (RealGroup.rows p) (RealGroup.columns p) p.GM tile
def column (p : Parameters) (tile : Nat) :=
  Schedule.column (RealGroup.rows p) (RealGroup.columns p) p.GM tile

def prepared (p : Parameters) (s : BlockState) (tile : Nat) : BlockState :=
  RealOffsets.result p (RealCoordinates.result p (RealGroup.result p s tile) tile)
    (row p tile) (column p tile)

theorem partition (p : Parameters) : RealGroup.setup p =
    RealGroup.code p ++ RealCoordinates.code p ++ RealOffsets.code p := by
  simp only [RealGroup.setup, RealGroup.code, RealCoordinates.code, RealOffsets.code, RealProgram.body, RealSource.kernel,
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
  rfl

/-- Composition of all fourteen actual tile-setup statements. It establishes
the grouped output coordinates and the exact clamped original input lanes. -/
theorem prepares (p : Parameters) (s : BlockState) (tile : Nat)
    (hGM : 0 < p.GM) (htile : tile < RealGroup.rows p*RealGroup.columns p)
    (ht : s.regs .int [] "tile_id" = some (Tile.scalar ((tile : Int)-(p.SMS : Int))))
    (hr : s.regs .nat [] "num_pid_m" = some (Tile.scalar (RealGroup.rows p)))
    (hg : s.regs .nat [] "num_pid_in_group" = some (Tile.scalar (p.GM*RealGroup.columns p))) :
    stepStmts (RealGroup.setup p) s = some (prepared p s tile) := by
  have h1 := RealGroup.executes p s tile hGM htile ht hr hg
  have h2 := RealCoordinates.executes p (RealGroup.result p s tile) tile
    (by simp [RealGroup.result]) (by simp [RealGroup.result])
    (by simp [RealGroup.result]) (by simp [RealGroup.result,hg])
  have h3 := RealOffsets.executes p
    (RealCoordinates.result p (RealGroup.result p s tile) tile)
    (row p tile) (column p tile)
    (by simp [RealCoordinates.result,row]) (by simp [RealCoordinates.result,column])
  rw [partition, List.append_assoc, stepStmts.append_some h1, stepStmts.append_some h2]
  exact h3

theorem prepared_frame (p : Parameters) (s : BlockState) (tile : Nat) :
    {prepared p s tile with regs := s.regs} = s := rfl

def condition : Op .bool [] :=
  .eq .int .nil (.ref .int [] "ki") (.constInt 0)

def result (p : Parameters) (s : BlockState) (tile ki : Nat) : BlockState :=
  let t := s.setReg "ki" .int [] (Tile.scalar (ki : Int))
  if ki = 0 then prepared p t tile else t

theorem matches_source (p : Parameters) : RealProgram.schedule p =
    [RealControl.update, .ifThen condition (RealGroup.setup p)] := by
  simp only [RealProgram.schedule, RealControl.update, condition, RealGroup.setup,
    RealProgram.body, RealSource.kernel, Parameters.source, TypedSource.kernel,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, List.map_cons, List.map_nil,
    FloatDType.toTileDType, List.getElem!_cons_zero, List.getElem!_cons_succ,
    RealProgram.loopBody, RealProgram.thenBody, List.take, List.drop]

/-- Both branches of the actual scheduling prefix: a signed phase update,
followed by all fourteen group/clamp statements exactly at a K restart. -/
theorem executes (p : Parameters) (s : BlockState) (tile ki count : Nat) (old : Int)
    (hGM : 0 < p.GM) (htile : tile < RealGroup.rows p*RealGroup.columns p)
    (ho : s.regs .int [] "ki" = some (Tile.scalar old))
    (hc : s.regs .nat [] "k_tiles" = some (Tile.scalar count))
    (hnext : RealControl.nextK old count = (ki : Int))
    (ht : ki=0 → s.regs .int [] "tile_id" = some (Tile.scalar ((tile : Int)-(p.SMS : Int))))
    (hr : s.regs .nat [] "num_pid_m" = some (Tile.scalar (RealGroup.rows p)))
    (hg : s.regs .nat [] "num_pid_in_group" = some (Tile.scalar (p.GM*RealGroup.columns p))) :
    stepStmts (RealProgram.schedule p) s = some (result p s tile ki) := by
  have hfirst := RealControl.executes p s old count ho hc
  rw [RealControl.matches_source,hnext] at hfirst
  have hupdate : stepStmt RealControl.update s =
      some (s.setReg "ki" .int [] (Tile.scalar (ki : Int))) := by
    cases hh : stepStmt RealControl.update s <;> simpa [stepStmts,hh] using hfirst
  have hcond : evalOp condition (s.setReg "ki" .int [] (Tile.scalar (ki : Int))) =
      some (Tile.scalar (decide (ki=0))) := by
    simp [condition, evalOp.eq_def, Tile.scalar, Tile.cop, ComparableDType.eq,
      Broadcast.leftIndex, Broadcast.rightIndex, bind, Option.bind, -Tile.scalar_eta]
  have hsecond : stepStmt (.ifThen condition (RealGroup.setup p))
      (s.setReg "ki" .int [] (Tile.scalar (ki : Int))) = some (result p s tile ki) := by
    rw [stepStmt,hcond]
    simp only [bind,Option.bind_some,Tile.scalar]
    by_cases hz : ki=0
    · simp only [hz,decide_true,if_true,result]
      exact prepares p (s.setReg "ki" .int [] (Tile.scalar (0 : Int))) tile hGM htile
        (by simpa using ht hz) (by simpa using hr) (by simpa using hg)
    · simp only [hz,decide_false,Bool.false_eq_true,if_false,result]
      rfl
  rw [matches_source]
  simp only [stepStmts,hupdate,hsecond]

theorem frame (p : Parameters) (s : BlockState) (tile ki : Nat) :
    {result p s tile ki with regs := s.regs} = s := by
  unfold result
  split <;> rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSchedule
