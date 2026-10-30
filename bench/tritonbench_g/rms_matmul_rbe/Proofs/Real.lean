import bench.tritonbench_g.rms_matmul_rbe.Proofs.Reference
import bench.tritonbench_g.rms_matmul_rbe.Proofs.RmsMatmulRbe_Real

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open RmsMatmulRbe (Parameters RowInput DotPlan Reduction steps)
noncomputable section
set_option maxHeartbeats 4000000

/-- Ideal arithmetic is an independent interpretation. Native FP32 words,
storage quantization, FTZ behavior and scalar overflow are not Real equality. -/
def realOps : Ops ℝ where
  toOps := RmsMatmulRbe.realOps
  sub := (·-·)
  powi := fun x n => x^n
  literalNat := Nat.cast

/-- Standalone mathematical target: sum_j X*RMS*W divided by the RMS
denominator, followed by adjacent-pair head-local rotary on Q/K only. -/
def realTarget (p : Parameters ℝ) (x : RowInput ℝ) (rotary : Bool) (row n : Nat) : ℝ :=
  RmsMatmulRbe.realTarget p x rotary row n

theorem real_accumulation (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → DotPlan) (n t : Nat) :
    accumulation realOps p x plans n t=
      RmsMatmulRbe.accumulation RmsMatmulRbe.realOps p x plans n t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change accumulation realOps p x plans n t +
      RmsMatmulRbe.dotBlock RmsMatmulRbe.realOps p x (plans t) n t 0 =
      RmsMatmulRbe.dotBlock RmsMatmulRbe.realOps p x (plans t) n t
        (RmsMatmulRbe.accumulation RmsMatmulRbe.realOps p x plans n t)
    rw [ih]
    rw [RmsMatmulRbe.dotBlock_real,RmsMatmulRbe.dotBlock_real]
    simp only [zero_add]

theorem real_moment (p : Parameters ℝ) (x : RowInput ℝ) (l : Fin 64) (t : Nat) :
    moment realOps p x l t=RmsMatmulRbe.moment RmsMatmulRbe.realOps p x l t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change moment realOps p x l t +
      (RmsMatmulRbe.masked RmsMatmulRbe.realOps p x.x t l)^(2:Int) =
      RmsMatmulRbe.moment RmsMatmulRbe.realOps p x l t +
        RmsMatmulRbe.square RmsMatmulRbe.realOps
          (RmsMatmulRbe.masked RmsMatmulRbe.realOps p x.x t l)
    rw [ih]
    simp [RmsMatmulRbe.square,RmsMatmulRbe.realOps,pow_two,zpow_ofNat]

theorem real_norm (p : Parameters ℝ) (x : RowInput ℝ) (r : Reduction) :
    norm realOps p x r=RmsMatmulRbe.realNorm p x := by
  unfold norm
  simp_rw [real_moment]
  exact RmsMatmulRbe.norm_real p x r

theorem real_projection (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (n : Nat) :
    projection realOps p x plans r n=RmsMatmulRbe.realProjection p x n := by
  unfold projection
  rw [real_accumulation,real_norm,RmsMatmulRbe.projection_sum]
  rfl

theorem value_real (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (rotary : Bool) (row n : Nat) :
    value realOps p x plans r rotary row n=realTarget p x rotary row n := by
  have pair : n/2*2=n-n%2 := by omega
  unfold value
  simp_rw [real_projection]
  simp only [pair,angle,realTarget,
    RmsMatmulRbe.realTarget,realOps,RmsMatmulRbe.realOps,Int.cast_add,Int.cast_natCast]

theorem reference_real (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (rotary : Bool) (row n : Nat) :
    referenceValue realOps p x plans r rotary row n=realTarget p x rotary row n := by
  rw [←value_reference,value_real]

#axiomsClean real_accumulation
#axiomsClean real_moment
#axiomsClean value_real
#axiomsClean reference_real
#specNonCircular realTarget avoiding [value,referenceValue,accumulation,moment,projection]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
