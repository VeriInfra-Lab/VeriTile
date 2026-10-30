import bench.repairs.tritonbench_g.masked_add_cuda.Kernels
import bench.tritonbench_g.masked_add_cuda.Proofs.Source
import bench.tritonbench_g.vector_addition.Proofs.Value

namespace VeriTile.Bench.Repairs.MaskedAddCuda
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters

def parameters (grad data mask : RegionName) (n : Nat) : Parameters :=
  ⟨data, mask, grad, n, blockSize⟩

abbrev candidateKernel (storage : MaskStorage) (p : Parameters) (alpha : ℝ) :=
  kernel storage p.output p.x p.y p.n alpha p.block

private theorem erased (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgKernel_fromKernelBody]

/-- Complete mathematical algorithm connection. The external signature adds
all three actual input ports, including the signed mask. The old helper's
extra same-format alpha cast is erased only in this Real projection. -/
theorem source_algorithm (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (kernel storage g p m n a B).eraseDType.toAlgKernel =
      { (VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel
          storage Bool.false g p m n a B).eraseDType.toAlgKernel with inputs := [g,p,m] } := by
  cases storage
  all_goals
    rw [erased _ (by rfl), erased _ (by rfl)]
    simp only [kernel,commutedKernel,source32,source64,commuted32,commuted64,
      VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
      Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
    simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
    try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComputeDType.eraseDType]
    rfl

theorem source_execution (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : BlockState) :
    exec (kernel storage g p m n a B).eraseDType.toAlgKernel s =
      exec (VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel
          storage Bool.false g p m n a B).eraseDType.toAlgKernel s := by
  rw [source_algorithm]
  rfl

/-- Complete mathematical algorithm connection. The external signature adds
all three actual input ports, including the signed mask. The old helper's
extra same-format alpha cast is erased only in this Real projection. -/
theorem commuted_algorithm (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (commutedKernel storage g p m n a B).eraseDType.toAlgKernel =
      { (VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel
          storage Bool.true g p m n a B).eraseDType.toAlgKernel with inputs := [g,p,m] } := by
  cases storage
  all_goals
    rw [erased _ (by rfl), erased _ (by rfl)]
    simp only [kernel,commutedKernel,source32,source64,commuted32,commuted64,
      VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
      Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
    simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
    try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComputeDType.eraseDType]
    rfl

theorem commuted_execution (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : BlockState) :
    exec (commutedKernel storage g p m n a B).eraseDType.toAlgKernel s =
      exec (VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel
          storage Bool.true g p m n a B).eraseDType.toAlgKernel s := by
  rw [commuted_algorithm]
  rfl

/-- Successful projection of every complete selected native body. -/
theorem source_supported (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (kernel storage g p m n a B).eraseDType.toAlgorithm? =
      .ok (kernel storage g p m n a B).eraseDType.toAlgKernel := by
  have h : (kernel storage g p m n a B).toAlgorithm? =
      .ok (kernel storage g p m n a B).toAlgKernel := by cases storage <;> rfl
  simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

/-- Whole mathematical program, including its complete external port list. -/
theorem source_projection (storage : MaskStorage) (p : Parameters) (alpha : ℝ) :
    (candidateKernel storage p alpha).eraseDType.toAlgKernel =
      { (VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_math_kernel
        p.output p.x p.y p.n alpha p.block).toAlgKernel with
        inputs := [p.output,p.x,p.y] } := by
  rw [candidateKernel,source_algorithm,
    VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_projection]
  rfl

theorem signed_mask_channel (storage : MaskStorage) (p : Parameters) (a : ℝ) :
    (candidateKernel storage p a).surfaceBody[4]?.map
      VeriTile.Bench.TritonBenchG.MaskedAddCuda.integerMaskRead = some Bool.true := by
  cases storage <;> rfl

theorem fp32_arithmetic (storage : MaskStorage) (p : Parameters) (a : ℝ) :
    (candidateKernel storage p a).surfaceBody[8]?.map
      (VeriTile.Bench.TritonBenchG.MaskedAddCuda.arithmeticPrecision .fp32) = some Bool.true := by
  cases storage <;> rfl

#axiomsClean source_projection
#axiomsClean source_algorithm
#axiomsClean commuted_algorithm
#axiomsClean source_execution
#axiomsClean commuted_execution
#axiomsClean source_supported
#axiomsClean signed_mask_channel
#axiomsClean fp32_arithmetic
end VeriTile.Bench.Repairs.MaskedAddCuda
