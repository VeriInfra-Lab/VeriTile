import bench.tritonbench_g.masked_add_cuda.Sources.Mathematical_Kernels

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton

/-- Real projection of the shared repair, retaining the signed mask channel. -/
def integer_math_kernel (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) : ComputeKernel :=
  (VeriTile.Bench.Repairs.MaskedAddCuda.kernel .int64 g p m n a B).eraseDType

def integer_commuted_math_kernel (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) : ComputeKernel :=
  (VeriTile.Bench.Repairs.MaskedAddCuda.commutedKernel .int64 g p m n a B).eraseDType

/-- Signed source read and comparison to exactly zero, never a Bool-channel load. -/
def integerMaskRead : ComputeStmt → Bool
  | .assign .bool _ _ (.alg (.ne .int _ (.load .int _ _ _) (.broadcast (.constInt 0) _))) => Bool.true
  | _ => Bool.false

example (storage : MaskStorage) (v : Bool) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) :
    (integer_kernel storage v grad p mask n alpha B).surfaceBody[4]?.map integerMaskRead =
      some Bool.true := by
  cases storage <;> cases v <;> rfl

/-- Inspect the precision wrapper of the final arithmetic assignment. -/
def arithmeticPrecision (precision : ComputeDType) : ComputeStmt → Bool
  | .assign _ _ _ (.compute (.alg actual _)) => decide (actual = precision)
  | _ => Bool.false

example (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).surfaceBody[8]?.map
      (arithmeticPrecision .fp32) = some Bool.true := by
  cases storage <;> cases v <;> rfl

/-- The candidate keeps every original operation before the final addition. -/
example (storage : MaskStorage) (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage Bool.false g p m n a B).surfaceBody.take 8 =
      (integer_kernel storage Bool.true g p m n a B).surfaceBody.take 8 := by
  cases storage <;> rfl

private theorem typed_projects (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).toAlgorithm? =
      .ok (integer_kernel storage v g p m n a B).toAlgKernel := by
  cases storage <;> cases v <;> rfl

private theorem erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

theorem integer_projection (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).eraseDType.toAlgKernel =
      (if v then integer_commuted_math_kernel g p m n a B else
        integer_math_kernel g p m n a B).toAlgKernel := by
  rw [erased_kernel _ (typed_projects ..)]
  cases storage <;> cases v <;> simp only [Bool.false_eq_true, if_true, if_false]
  all_goals simp only [integer_kernel, integer_math_kernel, integer_commuted_math_kernel, VeriTile.Bench.Repairs.MaskedAddCuda.kernel, VeriTile.Bench.Repairs.MaskedAddCuda.commutedKernel, ComputeKernel.eraseDType, Bool.false_eq_true, Bool.true_eq, ↓reduceIte, ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  all_goals simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComputeDType.eraseDType]
  all_goals rfl

theorem integer_supported (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).eraseDType.toAlgorithm? =
      .ok (integer_kernel storage v g p m n a B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

private theorem fp64_erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

private theorem full_erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

end VeriTile.Bench.TritonBenchG.MaskedAddCuda
