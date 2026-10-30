import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.FiniteReference
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReal
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger.Logical (sum)
noncomputable section
open scoped BigOperators

/-- Reuse the established total Real primitives; the selected source reaches
only the operators fixed by Logical.algebra. Storage precision stays FP32 in IO. -/
def realAlgebra : Algebra ℝ := Logical.algebra FastLayernorm.KernelMath.algebra

def vector {n : Nat} (x : TileIndex [n] → ℝ) : Nat → ℝ := extend (fun i => x (i,()))
@[simp] theorem vector_index {n : Nat} (x : TileIndex [n] → ℝ) (i : TileIndex [n]) :
    vector x i.1.val=x i := by
  cases i with | mk i unit => cases unit; exact extend_fin _ i

def forwardMathInput (p : ForwardParameters) (v : ForwardValues ℝ p) (i : Nat) : ℝ :=
  vector (forwardX v) i+if p.hasResidual then vector (forwardResidual v) i else 0
def forwardMath (p : ForwardParameters) (v : ForwardValues ℝ p) : ChannelValues ℝ (forwardOutputs p) :=
  forwardResult
    (fun i => Logical.y p.columns p.rms p.hasBias p.epsilon (forwardMathInput p v)
      (vector (forwardWeight v)) (vector (forwardBias v)) i.1.val)
    (fun i => forwardMathInput p v i.1.val)
    (Logical.mean p.columns p.rms (forwardMathInput p v))
    (Logical.rstd p.columns p.rms p.epsilon (forwardMathInput p v))

def backwardMathHat (p : BackwardParameters) (v : BackwardValues ℝ p) (row col : Nat) : ℝ :=
  (vector (backwardX v row) col-if p.rms then 0 else backwardMean v row)*backwardInverse v row
def backwardMathDX (p : BackwardParameters) (v : BackwardValues ℝ p) (row col : Nat) : ℝ :=
  Logical.dx p.columns p.rms p.hasDres (vector (backwardX v row)) (vector (backwardWeight v))
    (vector (backwardDY v row)) (vector (backwardDres v row)) (backwardMean v row) (backwardInverse v row) col
def backwardMathY (p : BackwardParameters) (v : BackwardValues ℝ p) (row col : Nat) : ℝ :=
  backwardMathHat p v row col*vector (backwardWeight v) col+
    if p.hasBias then vector (backwardBias v) col else 0
def backwardMathDW (p : BackwardParameters) (pid : Nat) (v : BackwardValues ℝ p) (col : Nat) : ℝ :=
  ∑ row ∈ Finset.Ico (rowStart p pid) (rowStop p pid),
    vector (backwardDY v row) col*backwardMathHat p v row col
def backwardMathDB (p : BackwardParameters) (pid : Nat) (v : BackwardValues ℝ p) (col : Nat) : ℝ :=
  ∑ row ∈ Finset.Ico (rowStart p pid) (rowStop p pid),vector (backwardDY v row) col
def backwardMath (p : BackwardParameters) (pid : Nat) (v : BackwardValues ℝ p) :
    ChannelValues ℝ (backwardOutputs p) :=
  backwardResult (fun row i => backwardMathDX p v row i.1.val)
    (fun row i => backwardMathY p v row i.1.val)
    (fun i => backwardMathDW p pid v i.1.val) (fun i => backwardMathDB p pid v i.1.val)

namespace MathCongruence
theorem sum_eq (n : Nat) (x y : Nat → ℝ) (equal : ∀ i,i<n → x i=y i) : sum n x=sum n y :=
  Finset.sum_congr rfl (fun i hi => equal i (Finset.mem_range.mp hi))
theorem mean_eq (n : Nat) (rms : Bool) (x y : Nat → ℝ) (equal : ∀ i,i<n → x i=y i) :
    Logical.mean n rms x=Logical.mean n rms y := by
  unfold Logical.mean
  rw [sum_eq n x y equal]
theorem inverse_eq (n : Nat) (rms : Bool) (eps : ℝ) (x y : Nat → ℝ)
    (equal : ∀ i,i<n → x i=y i) : Logical.rstd n rms eps x=Logical.rstd n rms eps y := by
  have variance : Logical.variance n rms x=Logical.variance n rms y := by
    unfold Logical.variance
    rw [mean_eq n rms x y equal]
    congr 1
    apply sum_eq
    intro i hi
    rw [equal i hi]
  rw [Logical.rstd,Logical.rstd,variance]
theorem y_eq (n : Nat) (rms bias : Bool) (eps : ℝ) (x w b X W B : Nat → ℝ)
    (hx : ∀ i,i<n → x i=X i) (hw : ∀ i,i<n → w i=W i)
    (hb : bias=Bool.true → ∀ i,i<n → b i=B i) (i : Nat) (hi : i<n) :
    Logical.y n rms bias eps x w b i=Logical.y n rms bias eps X W B i := by
  unfold Logical.y Logical.normalized
  rw [hx i hi,hw i hi,mean_eq n rms x X hx,inverse_eq n rms eps x X hx]
  cases h : bias
  · simp only [h,Bool.false_eq_true,if_false]
  · simp only [h,if_true,hb h i hi]

theorem dx_eq (n : Nat) (rms extra : Bool) (x w dy dr X W DY DR : Nat → ℝ)
    (mu inv MU INV : ℝ) (hx : ∀ i,i<n → x i=X i) (hw : ∀ i,i<n → w i=W i)
    (hdy : ∀ i,i<n → dy i=DY i) (hdr : extra=Bool.true → ∀ i,i<n → dr i=DR i)
    (hmu : rms=Bool.false → mu=MU) (hinv : inv=INV) (i : Nat) (hi : i<n) :
    Logical.dx n rms extra x w dy dr mu inv i=Logical.dx n rms extra X W DY DR MU INV i := by
  have hat (j : Nat) (hj : j<n) :
      (x j-if rms then 0 else mu)*inv=(X j-if rms then 0 else MU)*INV := by
    rw [hx j hj,hinv]
    cases h : rms
    · simp only [h,Bool.false_eq_true,if_false,hmu h]
    · simp only [h,if_true]
  have g (j : Nat) (hj : j<n) : w j*dy j=W j*DY j := by rw [hw j hj,hdy j hj]
  have product : sum n (fun j => ((x j-if rms then 0 else mu)*inv)*(w j*dy j))=
      sum n (fun j => ((X j-if rms then 0 else MU)*INV)*(W j*DY j)) := by
    apply sum_eq
    intro j hj
    rw [hat j hj,g j hj]
  unfold Logical.dx
  dsimp only
  rw [product,sum_eq n _ _ g,hat i hi,g i hi,hinv]
  cases h : extra
  · simp only [h,Bool.false_eq_true,if_false]
  · simp only [h,if_true,hdr h i hi]
end MathCongruence

#specNonCircular forwardMath avoiding [forwardKernel,FP.Structural.exec,Forward.reference,forwardGraph]
#specNonCircular backwardMath avoiding [backwardKernel,FP.Structural.exec,Backward.reference,backwardGraph]
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
