import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFinish
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopProgram

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealVariantSource
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 20000

def rowOp (p : Parameters) (v : Bool) : Op .int [p.BM] :=
  .add .int .scalarL
    (if v then .mul .int .nil (.constInt (p.BM : Int)) (.ref .int [] "pid_m")
     else .mul .int .nil (.ref .int [] "pid_m") (.constInt (p.BM : Int)))
    (.castNatToInt (.arange p.BM))

def rowStatement (p : Parameters) (v : Bool) : Stmt :=
  .assign .int [p.BM] "offs_cm" (rowOp p v)

def store (p : Parameters) (v : Bool) : List Stmt :=
  rowStatement p v :: (RealStore.code p).drop 1

def body (p : Parameters) (v : Bool) : List Stmt :=
  (RealProgram.body p).take 8 ++ [.ifThen RealFinish.condition (store p v)]

def loop (p : Parameters) (v : Bool) : Stmt :=
  .forRangeDyn "_i" (.constNat 0) RealLoopProgram.bound (.constNat 1) (body p v)

/-- The original/candidate complete real programs differ only at the signed
integer output-row product. Their actual loop bounds and all other code agree. -/
theorem matches_source (p : Parameters) (v : Bool) :
    (RealSource.kernel p v).body = RealPrefix.code p ++ [loop p v] := by
  cases v <;>
    simp only [loop,body,store,rowStatement,rowOp,RealFinish.condition,RealPrefix.code,
    RealLoopProgram.bound,RealStore.code,RealProgram.body,RealSource.kernel,
    Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType]
  all_goals rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealVariantSource
