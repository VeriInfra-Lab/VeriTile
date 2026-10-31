import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_Erasure
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSource
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

/-- Complete static prefix: indices/mask, two transition loads and four
zero accumulators. Every fragment is extracted from the full source below. -/
def prelude (l : RegionName) (batch dim B : Nat) : List Stmt :=
  (Erasure.complexBackward Bool.false "s" l "y" "gs" "gx" "gl" "gy" 0 batch dim B).toAlgKernel.body.take 9

def body (s y gx gy : RegionName) (length batch dim B : Nat) : List Stmt :=
  match (Erasure.complexBackward Bool.false s "l" y "gs" gx "gl" gy length batch dim B).toAlgKernel.body.drop 9 with
  | .forLoop _ _ loop :: _ => loop
  | _ => []

def postlude (gs gl : RegionName) (B : Nat) : List Stmt :=
  (Erasure.complexBackward Bool.false "s" "l" "y" gs "gx" gl "gy" 0 0 0 B).toAlgKernel.body.drop 10

theorem source_body (s l y gs gx gl gy : RegionName) (length batch dim B : Nat) :
    (Erasure.complexBackward Bool.false s l y gs gx gl gy length batch dim B).toAlgKernel.body =
      prelude l batch dim B ++ [.forLoop "i" length (body s y gx gy length batch dim B)] ++
        postlude gs gl B := by
  simp [prelude, body, postlude, Erasure.complexBackward,TypedSource.complexBackward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardIndexCommuted,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, FloatDType.toTileDType,
    VeriTile.Triton.eraseDType, Except.bind, bind]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSource
