import bench.tritonbench_g.dropout_triton.Proofs.Source
import bench.tritonbench_g.dropout_triton.Proofs.Mathematical_RealIO

/-! Source projection and reuse of the established regional execution proof.
Read-only inputs may share physical storage; no FlatAlloc is used by the
regional source result. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.RegionProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DropoutTriton
noncomputable section

/-- Exact mathematical body of the canonical typed source. -/
theorem original_body (n : Nat) (p : ℝ) :
    (kernel n p).eraseDType.toAlgKernel.body =
      (RealProof.original .int32 "x" "keep" "out" n blockSize p).toAlgKernel.body :=
  Erasure.body .fp32 .int32 Bool.false "x" "keep" "out" n blockSize p

theorem projection (swapped : Bool) (n B : Nat) (p : ℝ) :
    (TypedSource.source .fp32 .int32 swapped "x" "keep" "out" n B p).eraseDType.toAlgorithm? =
      .ok (TypedSource.source .fp32 .int32 swapped "x" "keep" "out" n B p).eraseDType.toAlgKernel := by
  cases swapped
  all_goals simp only [TypedSource.source,VeriTile.Bench.Repairs.DropoutTriton.sourceFamily,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#axiomsClean original_body
end
end VeriTile.Bench.Repairs.DropoutTriton.RegionProof
