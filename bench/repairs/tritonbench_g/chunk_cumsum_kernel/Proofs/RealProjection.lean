import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.RealRow

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def actual (input output : RegionName) (T BT : Nat) : Kernel :=
  (repairedKernel .fp32 .fp32 input output T BT).eraseDType.toAlgKernel

def reference (input output : RegionName) (T BT : Nat) : Kernel :=
  (kernel .fp32 .fp32 input output T BT).eraseDType.toAlgKernel

def mask (T BT : Nat) : MaskOpt .real [BT] :=
  .mask (.lt .int .scalarR (.ref .int [BT] "times") (.castNatToInt (.constNat T)))

def pointer (region : RegionName) (T BT : Nat) : Op .ptr [BT] :=
  .ptrAdd .scalarL (.ptrBase region)
    (.castIntToNat (.add .int .scalarL
      (.castNatToInt (.mul .nat .nil (.ref .nat [] "i_bh") (.constNat T)))
      (.ref .int [BT] "times")))

/-- Projection helper; the numerical statements are reused verbatim from the
legacy source. The theorem below checks it against the actual typed AST. -/
def actualBody (input output : RegionName) (T BT : Nat) : List Stmt :=
  [ .assign .int [BT] "times" (.add .int .scalarL
      (.mul .int .nil (.ref .int [] "i_t") (.constInt BT)) (.castNatToInt (.arange BT))),
    .assign .ptr [BT] "p_s" (pointer input T BT),
    .assign .ptr [BT] "p_o" (pointer output T BT),
    .assign .real [BT] "b_s" (.load .real (.ptr (.ref .ptr [BT] "p_s")) (mask T BT)) ] ++
  ((ccLoopBody input output T BT).drop 3).take 3 ++
  [.store .real [BT] (.ptr (.ref .ptr [BT] "p_o")) (.ref .real [BT] "b_o") (mask T BT)]

def prelude : List Stmt :=
  [.assign .nat [] "i_bh" (.programId 0),.assign .real [] "b_z" (.full [] (.const 0))]

theorem actual_body (input output : RegionName) (T BT : Nat) :
    (actual input output T BT).body=prelude ++
      [.forRangeInt "i_t" (.constInt 0) (.constInt ((T+BT-1)/BT : Nat)) (.constNat 1)
        (actualBody input output T BT)] := by
  simp only [actual,repairedKernel,ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,eraseDType_float]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComparableDType.eraseDType]
  rfl

theorem reference_body (input output : RegionName) (T BT : Nat) :
    (reference input output T BT).body=prelude ++
      [.forRangeDyn "i_t" (.constNat 0) (ccStopOp T BT) (.constNat 1) (ccLoopBody input output T BT)] := by
  rw [show (reference input output T BT).body=(chunk_cumsum_scalar_surface input output T BT).toAlgKernel.body
    from RealSource.original_body .fp32 .fp32 input output T BT]
  simp only [chunk_cumsum_scalar_surface,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,prelude,ccLoopBody,ccStopOp]
  rfl

#axiomsClean actual_body
#axiomsClean reference_body
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
