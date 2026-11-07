import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_Erasure
import VeriTile.Triton.Semantics.Integer
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSource
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

def prelude (l : RegionName) (length batch dim B : Nat) : List Stmt :=
  (Erasure.complexBackward Bool.false "s" l "y" "gs" "gx" "gl" "gy" length batch dim B).toAlgKernel.body.take 12

def body (s y gx gy : RegionName) (length batch dim B : Nat) : List Stmt :=
  match (Erasure.complexBackward Bool.false s "l" y "gs" gx "gl" gy length batch dim B).toAlgKernel.body.drop 12 with
  | .forRangeInt _ _ _ _ loop :: _ => loop
  | _ => []

def postlude (gs gl : RegionName) (B : Nat) : List Stmt :=
  (Erasure.complexBackward Bool.false "s" "l" "y" gs "gx" gl "gy" 0 0 0 B).toAlgKernel.body.drop 13

theorem source_body (s l y gs gx gl gy : RegionName) (length batch dim B : Nat) :
    (Erasure.complexBackward Bool.false s l y gs gx gl gy length batch dim B).toAlgKernel.body =
      prelude l length batch dim B ++
        [.forRangeInt "i" (.constInt 0) (.castNatToInt (.constNat length)) (.constNat 1) (body s y gx gy length batch dim B)] ++
          postlude gs gl B := by
  unfold prelude body postlude Erasure.complexBackward TypedSource.complexBackward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.complexBackwardKernel
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
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSource
