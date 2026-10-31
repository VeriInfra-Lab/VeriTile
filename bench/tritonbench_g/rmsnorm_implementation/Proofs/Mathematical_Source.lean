import bench.tritonbench_g.rmsnorm_implementation.Sources.Mathematical_Kernels
import bench.tritonbench_g.rmsnorm_implementation.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

/-- Exact precision erasure of the current source, with no copied function body. -/
def uncastMathKernel
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  (VeriTile.Bench.Repairs.RmsnormImplementation.kernel X W O
    sxb sxm sxk sw sob som sok N B eps).eraseDType

/-- Exact precision erasure of the retained address-order candidate. -/
def commutedMathKernel
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  (VeriTile.Bench.Repairs.RmsnormImplementation.indexCommutedKernel X W O
    sxb sxm sxk sw sob som sok N B eps).eraseDType

def mathKernel (v : Bool) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : Kernel :=
  if v then (commutedMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel
  else (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel

theorem typed_projects (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).toAlgorithm? =
      .ok (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel := by
  cases input <;> cases weight <;> cases output <;> cases v <;> rfl

theorem typed_projection (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel =
      mathKernel v X W O sxb sxm sxk sw sob som sok N B eps := by
  simp only [ComputeKernel.eraseDType, typed_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases weight <;> cases output <;> cases v
  all_goals simp only [typedKernel, originalTyped,VeriTile.Bench.Repairs.RmsnormImplementation.kernel, mathKernel, commutedMathKernel, VeriTile.Bench.Repairs.RmsnormImplementation.kernel,
    VeriTile.Bench.Repairs.RmsnormImplementation.indexCommutedKernel,
    RealCastView.ofKernel,RealCastView.castBindings,RealCastView.castBinding,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody,ComputeStmt.listToAlgorithm?_map_alg,
    ComputeKernel.surfaceBody,ComputeKernel.fromKernelBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComputeDType.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    List.map_cons,List.map_nil,Except.bind,bind,
    uncastMathKernel, VeriTile.Bench.Repairs.RmsnormImplementation.kernel,
    VeriTile.Bench.Repairs.RmsnormImplementation.indexCommutedKernel,
    RealCastView.ofKernel,RealCastView.castBindings,RealCastView.castBinding,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody,ComputeStmt.listToAlgorithm?_map_alg,
    ComputeKernel.surfaceBody,ComputeKernel.fromKernelBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComputeDType.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    List.map_cons,List.map_nil,Except.bind,bind, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.inputs, ComputeKernel.outputs,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType, List.take, List.drop, List.cons_append, List.nil_append]
  all_goals rfl

theorem typed_supported (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgorithm? =
      .ok (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
