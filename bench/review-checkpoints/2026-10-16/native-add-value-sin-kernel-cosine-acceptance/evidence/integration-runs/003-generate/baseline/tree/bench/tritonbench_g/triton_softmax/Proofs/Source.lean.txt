import bench.tritonbench_g.triton_softmax.Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def mathKernel (commuted : Bool) : Source :=
  if commuted then softmaxSource%[tl.float64, tl.float64, indexCommuted]
  else fun out inp si so n B =>
    ComputeKernel.mk [inp] [out] (softmax_kernel out inp si so n B).surfaceBody

theorem typed_projects (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel input output v out inp si so n B).toAlgorithm? =
      .ok (typedKernel input output v out inp si so n B).toAlgKernel := by
  cases input <;> cases output <;> cases v <;> rfl

theorem typed_projection (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel input output v out inp si so n B).eraseDType.toAlgKernel =
      (mathKernel v out inp si so n B).toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases output <;> cases v
  all_goals simp only [typedKernel, mathKernel, Bool.false_eq_true, ↓reduceIte, softmax_kernel,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typed_supported (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel input output v out inp si so n B).eraseDType.toAlgorithm? =
      .ok (typedKernel input output v out inp si so n B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

/-- Alias-based pointers retain the actual input/output metadata. -/
theorem io_signature (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel input output v out inp si so n B).inputs = [inp] ∧
      (typedKernel input output v out inp si so n B).outputs = [out] := by
  cases input <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax
