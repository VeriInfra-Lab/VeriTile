import bench.tritonbench_g.int_scaled_matmul.Proofs.RealInteger
import bench.tritonbench_g.int_scaled_matmul.Sources.LogicalSum
namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
open VeriTile.Triton FP.Structural FP.IntegerConversion Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def logicalProduct (p : Parameters) (s : State ℝ) (r c k : Nat) : Int :=
  wrap 8 ((s.mem p.input (r*p.inputRowStride+k*p.inputInnerStride)).read .int) *
  wrap 8 ((s.mem p.weight (k*p.weightInnerStride+c*p.weightColumnStride)).read .int)
def logicalDot (p : Parameters) (s : State ℝ) (r c : Nat) : Int :=
  wrap 32 (LogicalSum.sum (logicalProduct p s r c) p.inner)
def logicalScaled (p : Parameters) (s : State ℝ) (r c : Nat) : ℝ :=
  (logicalDot p s r c : ℝ) * (s.mem p.scale (r*p.scaleRowStride)).read .real
def logicalResult (p : Parameters) (s : State ℝ) (r c : Nat) : Int :=
  wrap 32 (truncate (logicalScaled p s r c))

theorem lane_product (p : Parameters) (s : State ℝ)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (i : CIndex p) (hi : active p s i) (k : Nat) (hk : k<p.steps) (t : Fin p.blockInner) :
    aValue p s k (i.1,t,PUnit.unit)*bValue p s k (t,i.2.1,PUnit.unit)=
    if k*p.blockInner+t.val<p.inner then
      logicalProduct p s (outputRow p s i) (outputColumn p s i) (k*p.blockInner+t.val) else 0 := by
  have hr : inputRow p s i.1=outputRow p s i := Nat.mod_eq_of_lt hi.1
  have hc : inputColumn p s i.2.1=outputColumn p s i := Nat.mod_eq_of_lt hi.2
  have hmask : (p.even=Bool.true ∨ t.val<p.inner-k*p.blockInner) ↔ k*p.blockInner+t.val<p.inner := by
    constructor
    · intro h
      cases h with
      | inl h => exact even_lane_in_bounds p he h k hk t
      | inr h => omega
    · intro h; exact Or.inr (by omega)
  simp only [aValue,bValue,inputInner,aOffset,bOffset,hr,hc,hmask]
  by_cases h : k*p.blockInner+t.val<p.inner
  · simp only [if_pos h]
    simp [logicalProduct,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  · simp [h,wrap]

theorem blockSum_logical (p : Parameters) (s : State ℝ)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (i : CIndex p) (hi : active p s i) (k : Nat) (hk : k<p.steps) :
    blockSum p s k i=LogicalSum.sum (fun t => if k*p.blockInner+t<p.inner then
      logicalProduct p s (outputRow p s i) (outputColumn p s i) (k*p.blockInner+t) else 0) p.blockInner := by
  simp only [blockSum,lane_product p s he i hi k hk]
  simpa only [Int.zero_add] using LogicalSum.fin_fold (fun t => if k*p.blockInner+t<p.inner then
    logicalProduct p s (outputRow p s i) (outputColumn p s i) (k*p.blockInner+t) else 0) p.blockInner 0

theorem exact_logical (p : Parameters) (s : State ℝ)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (i : CIndex p) (hi : active p s i) (n : Nat) (hn : n≤p.steps) :
    exactAccumulation p s n i=LogicalSum.sum (fun k => if k<p.inner then
      logicalProduct p s (outputRow p s i) (outputColumn p s i) k else 0) (n*p.blockInner) := by
  induction n with
  | zero => simp [exactAccumulation,LogicalSum.sum]
  | succ n ih =>
    simp only [exactAccumulation,ih (by omega),blockSum_logical p s he i hi n (by omega),
      Nat.succ_mul,LogicalSum.sum_add]

theorem accumulation_logical (p : Parameters) (s : State ℝ)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (i : CIndex p) (hi : active p s i) :
    accumulation p s p.steps i=logicalDot p s (outputRow p s i) (outputColumn p s i) := by
  rw [accumulation_value,exact_logical p s he i hi _ (by omega),LogicalSum.sum_padding,
    Nat.min_eq_left hcover]
  rfl

theorem scaled_logical (p : Parameters) (s : State ℝ)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (i : CIndex p) (hi : active p s i) :
    scaled p s i=logicalScaled p s (outputRow p s i) (outputColumn p s i) := by
  unfold scaled logicalScaled
  rw [accumulation_logical p s he hcover i hi]

theorem expected_logical (p : Parameters) (s : State ℝ)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (i : CIndex p) (hi : active p s i) :
    expected p s i=logicalResult p s (outputRow p s i) (outputColumn p s i) := by
  unfold expected logicalResult
  rw [scaled_logical p s he hcover i hi]

#axiomsClean lane_product
#axiomsClean blockSum_logical
#axiomsClean exact_logical
#axiomsClean accumulation_logical
#axiomsClean scaled_logical
#axiomsClean expected_logical
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
