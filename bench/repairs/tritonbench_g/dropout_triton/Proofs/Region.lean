import bench.repairs.tritonbench_g.dropout_triton.Proofs.Source
import bench.tritonbench_g.dropout_triton.Proofs.RealIO

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

/-- The existing index-prefix proof is independent of allocation: flattening
has no effect on the two scalar index statements. -/
theorem prefix_equal (B : Nat) (s : BlockState) :
    stepStmts (Candidate.indexPrefix Bool.false B) s =
      stepStmts (Candidate.indexPrefix Bool.true B) s := by
  have h := Candidate.prefix_flat (⟨"unused", [], fun _ => 0, fun _ => 0⟩ : FlatAlloc) B s
  simpa only [Candidate.indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp, Bool.false_eq_true, ↓reduceIte] using h

theorem candidate_exec (n : Nat) (p : ℝ) (s : BlockState) :
    exec (kernel n p).eraseDType.toAlgKernel s =
      exec (indexCommutedKernel n p).eraseDType.toAlgKernel s := by
  unfold exec
  change stepStmts (TypedSource.source .fp32 .int32 Bool.false "x" "keep" "out" n blockSize p).eraseDType.toAlgKernel.body s =
    stepStmts (TypedSource.source .fp32 .int32 Bool.true "x" "keep" "out" n blockSize p).eraseDType.toAlgKernel.body s
  simp only [Erasure.body]
  rw [Candidate.body .int32 Bool.false, Candidate.body .int32 Bool.true]
  simp only [ProgramRewrite.append, prefix_equal]

/-- Commuting the pure index prefix preserves the actual access trace. -/
theorem candidate_trace (n : Nat) (p : ℝ) (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafe bounds (kernel n p).eraseDType.toAlgKernel s ↔
      Kernel.TraceSafe bounds (indexCommutedKernel n p).eraseDType.toAlgKernel s := by
  unfold Kernel.TraceSafe
  change Stmt.TraceSafeList bounds
      (TypedSource.source .fp32 .int32 Bool.false "x" "keep" "out" n blockSize p).eraseDType.toAlgKernel.body s ↔
    Stmt.TraceSafeList bounds
      (TypedSource.source .fp32 .int32 Bool.true "x" "keep" "out" n blockSize p).eraseDType.toAlgKernel.body s
  simp only [Erasure.body]
  rw [Candidate.body .int32 Bool.false,Candidate.body .int32 Bool.true]
  simp only [Candidate.indexPrefix,Bool.false_eq_true,↓reduceIte,List.cons_append,List.nil_append,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt,stepStmt,evalOp_programId,evalOp_mul,
    evalOp_ref,evalOp_constNat,bind,Option.bind_some,BlockState.setReg_same]
  simp only [Tile.bop,NumericDType.mul,Nat.mul_comm]

theorem projection (swapped : Bool) (n B : Nat) (p : ℝ) :
    (TypedSource.source .fp32 .int32 swapped "x" "keep" "out" n B p).eraseDType.toAlgorithm? =
      .ok (TypedSource.source .fp32 .int32 swapped "x" "keep" "out" n B p).eraseDType.toAlgKernel := by
  cases swapped
  all_goals simp only [TypedSource.source,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#axiomsClean original_body
#axiomsClean prefix_equal
#axiomsClean candidate_exec
end
end VeriTile.Bench.Repairs.DropoutTriton.RegionProof
