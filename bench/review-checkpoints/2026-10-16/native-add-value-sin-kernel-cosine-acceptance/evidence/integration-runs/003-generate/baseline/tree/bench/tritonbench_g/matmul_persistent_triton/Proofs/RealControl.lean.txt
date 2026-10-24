import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealProgram
import bench.tritonbench_g.matmul_persistent_triton.Proofs.IterationSchedule

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealControl
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def nextK (old : Int) (count : Nat) : Int :=
  if old = (count : Int)-1 then 0 else old+1

def update : Stmt :=
  .assign .int [] "ki" (.where
    (.eq .int .nil (.ref .int [] "ki")
      (.sub .int .nil (.castNatToInt (.ref .nat [] "k_tiles")) (.constInt 1)))
    (.constInt 0) (.add .int .nil (.ref .int [] "ki") (.constInt 1)))

theorem matches_source (p : Parameters) : (RealProgram.body p).take 1 = [update] := by
  simp only [RealProgram.body, RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType]
  rfl

/-- The real execution of the original signed where-update is total when its
actual two scalar registers are present. -/
theorem executes (p : Parameters) (s : BlockState) (old : Int) (count : Nat)
    (ho : s.regs .int [] "ki" = some (Tile.scalar old))
    (hc : s.regs .nat [] "k_tiles" = some (Tile.scalar count)) :
    stepStmts ((RealProgram.body p).take 1) s =
      some (s.setReg "ki" .int [] (Tile.scalar (nextK old count))) := by
  rw [matches_source]
  simp [update, nextK, stepStmts, stepStmt, evalOp.eq_def, ho, hc, bind, Option.bind,
    Tile.scalar, Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.sub,
    ComparableDType.eq, Broadcast.leftIndex, Broadcast.rightIndex, -Tile.scalar_eta]

/-- Negative initialization reaches phase zero at the first actual source
iteration whenever K has at least one block. -/
theorem initial (count : Nat) (hc : 0 < count) : nextK (-1) count = 0 :=
  Schedule.first_ki count hc

/-- Every subsequent actual update tracks the flat source-loop index modulo
the ceiling-divided K extent. -/
theorem next (count i : Nat) (hc : 0 < count) :
    nextK (i % count : Nat) count = ((i+1)%count : Nat) :=
  Schedule.ki_next count i hc

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealControl
