import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelValues
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Bounds

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelRealArithmetic
open VeriTile.Triton FP.Structural KernelSource KernelValues
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

@[simp] theorem decimal_zero : (0.0 : ℝ)=0 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num
@[simp] theorem decimal_one : (1.0 : ℝ)=1 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num
@[simp] theorem decimal_two : (2.0 : ℝ)=2 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num

/-- Pointwise lift of a finite inner-product evaluator. Batch dimensions are
preserved, although this source only reaches the unbatched case. -/
def laneDot {α : Type} (reduce : {k : Nat} → (Fin k → α) → (Fin k → α) → α) :
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) →
    TileIndex (batch++[m,n]) → α
  | [],_,_,_,a,b,i => reduce (fun j => a (i.1,j,PUnit.unit)) (fun j => b (j,i.2.1,PUnit.unit))
  | _::bs,_,_,_,a,b,i => laneDot reduce (batch:=bs) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2

def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | .max => max x y | .pow => 0
  unary := fun _ op x => match op with
    | .tanh => Real.tanh x | .erf => VeriTile.Math.realErf x | _ => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

def dot : Dot ℝ := ⟨fun _ _ _ _ _ => laneDot (fun a b => ∑ j, a j*b j)⟩

def x (mode : Mode) (p : Parameters) (s : State ℝ) (m k : Nat) : ℝ :=
  KernelValues.read mode s p.input (m*p.inputRowStride+k*p.inputInnerStride)
def w (mode : Mode) (p : Parameters) (s : State ℝ) (n k : Nat) : ℝ :=
  KernelValues.read mode s p.weight (n*p.weightColStride+k*p.weightInnerStride)
def bias (mode : Mode) (p : Parameters) (s : State ℝ) (n : Nat) : ℝ :=
  if p.hasBias then KernelValues.read mode s p.bias n else 0

/-- Independent dense mathematical target, indexed by logical row and column.
It does not use loop state, tiling, dot, source execution, or output memory. -/
def densePre (mode : Mode) (p : Parameters) (s : State ℝ) (m n : Nat) : ℝ :=
  bias mode p s n + ∑ k∈Finset.range p.inner, x mode p s m k*w mode p s n k

def denseOutput (mode : Mode) (p : Parameters) (s : State ℝ) (m n : Nat) : ℝ :=
  realActivation p.activation (densePre mode p s m n)

theorem initial_lane (mode : Mode) (p : Parameters) (s : State ℝ)
    (i : TileIndex [p.blockRows,p.blockColumns]) (hn : KernelValues.col p s i.2.1<p.columns) :
    initial algebra mode p s i=bias mode p s (KernelValues.col p s i.2.1) := by
  simp [initial,biasValue,zero,algebra,bias,Nat.mod_eq_of_lt hn,hn]

theorem mask_equiv (p : Parameters) (cur lane : Nat) (positive : 0<p.blockInner)
    (inside : cur<ceilDiv p.inner p.blockInner) (valid : lane<p.blockInner) :
    (p.inner%p.blockInner=0 ∨ (lane:Int)<(p.inner:Int)-(cur*p.blockInner:Int)) ↔
      cur*p.blockInner+lane<p.inner := by
  have sourceMask := source_k_mask p.inner p.blockInner cur lane positive inside valid
  have upper := ceil_upper p.inner p.blockInner positive
  have product := Nat.mul_le_mul_right p.blockInner (Nat.succ_le_of_lt inside)
  have before : cur*p.blockInner<p.inner := by nlinarith
  have compare : (lane:Int)<(p.inner:Int)-(cur*p.blockInner:Int) ↔ lane<p.inner-cur*p.blockInner := by
    omega
  rw [compare]
  exact sourceMask

theorem update_lane (mode : Mode) (p : Parameters) (s : State ℝ)
    (positive : 0<p.blockInner) (cur : Nat) (inside : cur<ceilDiv p.inner p.blockInner)
    (a : Accumulator ℝ p) (i : TileIndex [p.blockRows,p.blockColumns])
    (hm : KernelValues.row p s i.1<p.rows) (hn : KernelValues.col p s i.2.1<p.columns) :
    update algebra dot mode p s cur a i=a i+∑ k∈Finset.range p.blockInner,
      if cur*p.blockInner+k<p.inner then
        x mode p s (KernelValues.row p s i.1) (cur*p.blockInner+k)*
        w mode p s (KernelValues.col p s i.2.1) (cur*p.blockInner+k) else 0 := by
  simp only [KernelValues.update,algebra,dot,laneDot]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro j _
  simp only [aValues,bValues,mask_equiv p cur j.val positive inside j.isLt]
  by_cases activeK : cur*p.blockInner+j.val<p.inner
  · simp only [activeK,if_true,x,w,aOffset,bOffset,Nat.mod_eq_of_lt hm,Nat.mod_eq_of_lt hn]
    congr 2 <;> ring
  · simp [activeK,paddedZero,algebra]

theorem accumulated_lane (mode : Mode) (p : Parameters) (s : State ℝ)
    (positive : 0<p.blockInner) (i : TileIndex [p.blockRows,p.blockColumns])
    (hm : KernelValues.row p s i.1<p.rows) (hn : KernelValues.col p s i.2.1<p.columns) :
    accumulated algebra dot mode p s i=densePre mode p s (KernelValues.row p s i.1) (KernelValues.col p s i.2.1) := by
  let product := fun k => if k<p.inner then
    x mode p s (KernelValues.row p s i.1) k*w mode p s (KernelValues.col p s i.2.1) k else 0
  have prefixSum (count : Nat) (bound : count≤ceilDiv p.inner p.blockInner) :
      foldN (update algebra dot mode p s) count (initial algebra mode p s) i=
        bias mode p s (KernelValues.col p s i.2.1)+∑ k∈Finset.range (count*p.blockInner),product k := by
    induction count with
    | zero => simpa [foldN] using initial_lane mode p s i hn
    | succ count ih =>
      rw [foldN,update_lane mode p s positive count (by omega) _ i hm hn,ih (by omega)]
      rw [Nat.succ_mul,Finset.sum_range_add]
      simp only [product]
      ring
  unfold accumulated densePre
  rw [prefixSum _ (le_refl _)]
  congr 1
  have cover := ceil_cover p.inner p.blockInner positive
  calc
    _ = ∑ k∈Finset.range p.inner,product k := by
      symm
      apply Finset.sum_subset (Finset.range_mono cover)
      intro k _ outside
      have inactive : ¬k<p.inner := by simpa using outside
      simp [product,inactive]
    _ = _ := by
      apply Finset.sum_congr rfl
      intro k hk
      simp [product,Finset.mem_range.mp hk]

theorem activation_real (name : Option String) (value : ℝ) :
    KernelValues.activation algebra name value=realActivation name value := by
  unfold KernelValues.activation realActivation
  split_ifs <;> simp only [algebra,zero,decimal_one,decimal_two]
  · ring
  · have cubic : 0.044715*value*value*value=0.044715*value^3 := by ring
    rw [cubic]
    ring

#axiomsClean accumulated_lane
#axiomsClean activation_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelRealArithmetic
