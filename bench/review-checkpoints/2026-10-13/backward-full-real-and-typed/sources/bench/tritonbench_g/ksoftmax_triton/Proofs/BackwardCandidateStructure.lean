import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullErasure
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidatePrefix

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Candidate
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

private theorem erase_take (n : Nat) (body : List Stmt) :
    (Stmt.eraseDTypeList body).take n = Stmt.eraseDTypeList (body.take n) := by
  induction n generalizing body with
  | zero => simp only [List.take_zero, Stmt.eraseDTypeList]
  | succ n ih =>
    cases body with
    | nil => simp only [Stmt.eraseDTypeList, List.take_nil]
    | cons head rest =>
      simp only [Stmt.eraseDTypeList, List.take_succ_cons, ih]

private theorem erase_drop (n : Nat) (body : List Stmt) :
    (Stmt.eraseDTypeList body).drop n = Stmt.eraseDTypeList (body.drop n) := by
  induction n generalizing body with
  | zero => rfl
  | succ n ih =>
    cases body with
    | nil => simp only [Stmt.eraseDTypeList, List.drop_nil]
    | cons head rest =>
      simp only [Stmt.eraseDTypeList, List.drop_succ_cons, ih]

private theorem erased_body (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body =
    Stmt.eraseDTypeList (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).toAlgKernel.body := by
  simp only [ComputeKernel.eraseDType, full_typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody, Kernel.eraseDType]

/-- Before erasure, the original and candidate have literally the same
complete suffix. No load, branch, cast or store is replaced by a helper. -/
private theorem raw_suffix (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).toAlgKernel.body.drop 4 =
    (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).toAlgKernel.body.drop 4 := by
  cases gradient <;> cases saved <;> cases destination <;> cases v <;> rfl

private theorem erased_take (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body.take 4 =
      indexPrefix v GradOut gm gn D := by
  rw [erased_body, erase_take]
  cases gradient <;> cases saved <;> cases destination <;> cases v
  all_goals
    dsimp only [Typed.kernel,
      Typed.kernel_fp32_fp32, Typed.kernel_fp32_fp64, Typed.kernel_fp32_fp16,
      Typed.kernel_fp32_bf16, Typed.kernel_fp32_f8e4, Typed.kernel_fp32_f8e5,
      Typed.kernel_fp64_fp32, Typed.kernel_fp64_fp64, Typed.kernel_fp64_fp16,
      Typed.kernel_fp64_bf16, Typed.kernel_fp64_f8e4, Typed.kernel_fp64_f8e5,
      Typed.kernel_fp16_fp32, Typed.kernel_fp16_fp64, Typed.kernel_fp16_fp16,
      Typed.kernel_fp16_bf16, Typed.kernel_fp16_f8e4, Typed.kernel_fp16_f8e5,
      Typed.kernel_bf16_fp32, Typed.kernel_bf16_fp64, Typed.kernel_bf16_fp16,
      Typed.kernel_bf16_bf16, Typed.kernel_bf16_f8e4, Typed.kernel_bf16_f8e5,
      Typed.kernel_f8e4_fp32, Typed.kernel_f8e4_fp64, Typed.kernel_f8e4_fp16,
      Typed.kernel_f8e4_bf16, Typed.kernel_f8e4_f8e4, Typed.kernel_f8e4_f8e5,
      Typed.kernel_f8e5_fp32, Typed.kernel_f8e5_fp64, Typed.kernel_f8e5_fp16,
      Typed.kernel_f8e5_bf16, Typed.kernel_f8e5_f8e4, Typed.kernel_f8e5_f8e5,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, Except.pure, Bind.bind, Pure.pure, ComputeDType.eraseDType]
    -- Restrict to the four pure assignments before expanding any erasure.
    simp only [List.take_succ_cons, List.take_zero]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      FloatDType.toTileDType, Region.cast, indexPrefix]
    try rfl

/-- Exact decomposition of the actual erased full bodies, with a shared
original suffix. The proof erases only four assignments explicitly; generic
take/drop laws retain the entire suffix without expanding its control tree. -/
theorem typed_body (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body =
    indexPrefix v GradOut gm gn D ++
      (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body.drop 4 := by
  calc
    _ = (Typed.kernel gradient saved destination v GradIn GradOut Out
          bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body.take 4 ++
        (Typed.kernel gradient saved destination v GradIn GradOut Out
          bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body.drop 4 :=
      (List.take_append_drop 4 _).symm
    _ = _ := by
      rw [erased_take]
      congr 1
      simp only [erased_body, erase_drop, raw_suffix]

#axiomsClean typed_body
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Candidate
