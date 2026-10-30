import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmValues
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelScalarProfile

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
open VeriTile.Triton FP.Structural KernelExecution
open Int8DequantMatmul (rsum finRange_rsum)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000

/-- The same dimensions, complete autotune tile, loop count and launch
strides as the checked Python; K=0 retains its canonical inert stride. -/
def parameters (s : Shape) (c : Config) (columnMajor : Bool) : Parameters where
  input := "A"
  weight := "B"
  inputScale := "AS"
  weightScale := "BS"
  output := "C"
  rows := s.M
  columns := s.N
  inner := s.K
  inputRowStride := max s.K 1
  inputInnerStride := 1
  inputScaleStride := 1
  weightInnerStride := if columnMajor then 1 else max s.N 1
  weightColumnStride := if columnMajor then max s.K 1 else 1
  weightScaleStride := 1
  outputRowStride := max s.N 1
  outputColumnStride := 1
  blockRows := c.BM
  blockColumns := c.BN
  blockInner := c.BK
  groupRows := c.group
  split := c.S
  steps := steps s c

def realInput (s : Shape) (input : Input ℝ) : RealInput s where
  A := fun r k => ((input.A r k).toInt : ℝ)
  B := fun k n => ((input.B k n).toInt : ℝ)
  aScale := input.aScale
  bScale := input.bScale

theorem accumulator_real (s : Shape) (c : Config) (columnMajor : Bool) (plans : Plans)
    (input : Input ℝ) (r n z t : Nat) :
    (accumulator mathematicalIntegers KernelIntegerIdeal.operations.toOps plans (parameters s c columnMajor)
      input r n z t : ℝ) =
      rsum (fun t => rsum (fun lane => if sourceK c t z lane < s.K then
        (realInput s input).A r (sourceK c t z lane)*(realInput s input).B (sourceK c t z lane) n else 0) c.BK) t := by
  induction t with
  | zero => simp [accumulator, rsum]
  | succ t ih =>
    simp only [mathematicalIntegers] at ih
    simp only [accumulator, mathematicalIntegers, id_eq]
    rw [KernelIntegerIdeal.plan_cast_real, Int8DequantMatmul.DotPlan.real, ih]
    have products (lane : Fin c.BK) :
        Int8DequantMatmul.realProduct
          (if inner (parameters s c columnMajor) t z lane < s.K then
            input.A r (inner (parameters s c columnMajor) t z lane) else 0)
          (if inner (parameters s c columnMajor) t z lane < s.K then
            input.B (inner (parameters s c columnMajor) t z lane) n else 0) =
        if sourceK c t z lane < s.K then
          (realInput s input).A r (sourceK c t z lane)*(realInput s input).B (sourceK c t z lane) n else 0 := by
      change Int8DequantMatmul.realProduct
        (if sourceK c t z lane < s.K then input.A r (sourceK c t z lane) else 0)
        (if sourceK c t z lane < s.K then input.B (sourceK c t z lane) n else 0) = _
      by_cases hk : sourceK c t z lane < s.K <;>
        simp [Int8DequantMatmul.realProduct, hk, realInput]
    simp only [parameters] at products ⊢
    simp_rw [products]
    rw [finRange_rsum (fun lane => if sourceK c t z lane < s.K then
      (realInput s input).A r (sourceK c t z lane)*(realInput s input).B (sourceK c t z lane) n else 0) c.BK]
    rfl

theorem partial_real (s : Shape) (c : Config) (columnMajor : Bool) (plans : Plans)
    (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (rowScale : Precision)
    (input : Input ℝ) (r n z : Nat) :
    partialWord (KernelScalarProfile.mathematical tree rest) mathematicalIntegers
      KernelIntegerIdeal.operations.toOps plans rowScale (parameters s c columnMajor) input r n z =
      realPartial s c (realInput s input) r n z := by
  unfold partialWord
  simp only [KernelScalarProfile.mathematical]
  rw [accumulator_real]
  cases rowScale <;> rfl

/-- The independent dense formula is reached in the ideal interpretation,
for all split configurations, all tails and K=0. -/
theorem dense_formula (s : Shape) (c : Config) (columnMajor : Bool) (plans : Plans)
    (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (rowScale : Precision)
    (input : Input ℝ) (r n : Nat) :
    rsum (fun z => partialWord (KernelScalarProfile.mathematical tree rest) mathematicalIntegers
      KernelIntegerIdeal.operations.toOps plans rowScale (parameters s c columnMajor) input r n z) c.S =
      target s (realInput s input) r n := by
  simp_rw [partial_real]
  exact partials_target s c (realInput s input) r n

#axiomsClean accumulator_real
#axiomsClean partial_real
#axiomsClean dense_formula
#specNonCircular realInput avoiding [KernelGemm.source, KernelExecution.exec, accumulator]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
