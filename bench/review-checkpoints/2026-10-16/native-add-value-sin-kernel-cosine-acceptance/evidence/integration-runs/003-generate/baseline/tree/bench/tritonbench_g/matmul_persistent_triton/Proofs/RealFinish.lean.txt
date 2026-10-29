import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealStore

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealFinish
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def condition : Op .bool [] :=
  .eq .int .nil (.ref .int [] "ki")
    (.sub .int .nil (.castNatToInt (.ref .nat [] "k_tiles")) (.constInt 1))

theorem matches_source (p : Parameters) : RealProgram.store p =
    [.ifThen condition (RealStore.code p)] := by
  simp only [RealProgram.store, RealStore.code, RealProgram.body, RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType]
  rfl

theorem evaluates (s : BlockState) (ki count : Nat)
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hc : s.regs .nat [] "k_tiles" = some (Tile.scalar count)) :
    evalOp condition s = some (Tile.scalar (decide (ki+1=count))) := by
  have h : ((ki : Int) = (count : Int)-1) ↔ ki+1=count := by omega
  simp [condition,evalOp.eq_def,hk,hc,bind,Option.bind,Tile.scalar,Tile.bop,Tile.cop,
    NumericDType.sub,ComparableDType.eq,Broadcast.leftIndex,Broadcast.rightIndex,h,-Tile.scalar_eta]

/-- The actual source condition leaves memory and accumulator unchanged before
the last K block and executes every masked-store/reset statement at completion. -/
theorem executes (p : Parameters) (s : BlockState) (row column ki count : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ)
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hn : s.regs .nat [] "k_tiles" = some (Tile.scalar count))
    (hr : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hc : s.regs .int [] "pid_n" = some (Tile.scalar (column : Int)))
    (hv : s.regs .real [p.BM,p.BN] "accumulator" = some (Tile.mat fun i j => some (values i j))) :
    stepStmts (RealProgram.store p) s =
      some (if ki+1=count then RealStore.result p s row column values else s) := by
  rw [matches_source]
  simp only [stepStmts,stepStmt,evaluates s ki count hk hn,bind,Option.bind_some,Tile.scalar]
  by_cases h : ki+1=count
  · simp only [h,decide_true,if_true]
    rw [RealStore.executes p s row column values hr hc hv]
  · simp only [h,decide_false,Bool.false_eq_true,if_false]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealFinish
