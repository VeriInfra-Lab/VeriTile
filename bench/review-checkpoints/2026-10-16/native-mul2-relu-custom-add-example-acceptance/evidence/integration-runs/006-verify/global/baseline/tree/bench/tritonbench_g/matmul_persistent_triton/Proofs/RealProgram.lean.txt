import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealSource
import VeriTile.Triton.Semantics.Step

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealProgram
open VeriTile.Triton
set_option maxHeartbeats 14000000
set_option maxRecDepth 16000
noncomputable section
local instance : Inhabited Stmt := ⟨.assign .nat [] "" (.constNat 0)⟩
def loopBody : Stmt → List Stmt
  | .forRangeDyn _ _ _ _ body => body
  | _ => []
def thenBody : Stmt → List Stmt
  | .ifThen _ body | .ifThenElse _ body _ => body
  | _ => []
def body (p : Parameters) : List Stmt :=
  loopBody (RealSource.kernel p Bool.false).body[16]!
def schedule (p : Parameters) : List Stmt := (body p).take 2
def loads (p : Parameters) : List Stmt := ((body p).drop 2).take 5
def dot (p : Parameters) : List Stmt := ((body p).drop 7).take 1
def store (p : Parameters) : List Stmt := ((body p).drop 8).take 1

/-- Source fragments partition the one genuine flat persistent loop body. -/
theorem partition (p : Parameters) : body p =
    schedule p ++ loads p ++ dot p ++ store p := by
  simp only [body, schedule, loads, dot, store, RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType]
  rfl
end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealProgram
