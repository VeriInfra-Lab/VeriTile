import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_Erasure
import VeriTile.Triton.Semantics.Integer
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSource
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

/-- Complete actual prefix, including the three signed shape casts. -/
def prelude (s l : RegionName) (length batch dim block : Nat) : List Stmt :=
  (Erasure.complexForward Bool.false s "input" "output" l length batch dim block).toAlgKernel.body.take 10

/-- Complete actual signed loop body, including both stores and simultaneous update. -/
def body (x y : RegionName) (batch dim block : Nat) : List Stmt :=
  match (Erasure.complexForward Bool.false "initial" x y "transition" 0 batch dim block).toAlgKernel.body.drop 10 with
  | [.forRangeInt _ _ _ _ loop] => loop
  | _ => []

theorem source_body (s x y l : RegionName) (length batch dim block : Nat) :
    (Erasure.complexForward Bool.false s x y l length batch dim block).toAlgKernel.body =
      prelude s l length batch dim block ++
        [.forRangeInt "t" (.constInt 0) (.castNatToInt (.constNat length)) (.constNat 1) (body x y batch dim block)] := by
  unfold prelude body Erasure.complexForward TypedSource.complexForward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSource
