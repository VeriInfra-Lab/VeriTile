import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardModel
import bench.tritonbench_g.layer_norm_liger.Proofs.Logical
namespace VeriTile.Bench.Repairs.LayerNormOps.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger.Logical (sum sumValues padded_sum)
open scoped BigOperators
noncomputable section

/-- Independently stated population-normalization target. RMS omits mean
subtraction; the affine bias and residual are independent optional inputs. -/
def mean (n : Nat) (rms : Bool) (x : Nat → ℝ) := if rms then 0 else sum n x / n
def variance (n : Nat) (rms : Bool) (x : Nat → ℝ) :=
  sum n (fun i => (x i - mean n rms x)^2) / n
def rstd (n : Nat) (rms : Bool) (eps : ℝ) (x : Nat → ℝ) := 1 / Real.sqrt (variance n rms x + eps)
def normalized (n : Nat) (rms : Bool) (eps : ℝ) (x : Nat → ℝ) (i : Nat) :=
  (x i - mean n rms x) * rstd n rms eps x
def y (n : Nat) (rms bias : Bool) (eps : ℝ) (x w b : Nat → ℝ) (i : Nat) :=
  normalized n rms eps x i*w i + if bias then b i else 0

def dx (n : Nat) (rms extra : Bool) (x w dy dres : Nat → ℝ) (mu inv : ℝ) (i : Nat) :=
  let h := fun j => (x j - if rms then 0 else mu)*inv
  let g := fun j => w j*dy j
  (g i - h i*(sum n (fun j => h j*g j)/n) -
    (if rms then 0 else sum n g/n))*inv + (if extra then dres i else 0)

def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  { LayerNormLiger.Logical.algebra rest with
    unary := fun pr op x => match op with
      | .sqrt => Real.sqrt x
      | _ => (LayerNormLiger.Logical.algebra rest).unary pr op x }

def input (p : ForwardParameters) (s : State ℝ) (i : Nat) :=
  (s.mem p.x (s.pids 0*p.xStride+i)).read .fp32 +
    if p.hasResidual then (s.mem p.residual (s.pids 0*p.residualStride+i)).read .fp32 else 0

theorem z_value (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ) (i : TileIndex [p.block]) :
    Forward.z (algebra rest) p s i = if i.1.val < p.columns then input p s i.1.val else 0 := by
  cases ha : p.hasResidual <;> by_cases hi : i.1.val < p.columns <;>
    simp [Forward.z, Forward.loaded, Forward.cast, Forward.padding, Forward.literal, Forward.binary,
      algebra, LayerNormLiger.Logical.algebra, input, ha, hi]

theorem mean_value (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) :
    Forward.meanValue (algebra rest) p s = sum p.columns (input p s) / p.columns := by
  change (∑ i : Fin p.block, Forward.z (algebra rest) p s (i,PUnit.unit)) / p.columns = _
  simp_rw [z_value]
  rw [padded_sum _ _ hb]

theorem centered_value (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    Forward.centered (algebra rest) p s i = input p s i.1.val - mean p.columns p.rms (input p s) := by
  unfold Forward.centered
  rw [z_value, mean_value rest p s hb]
  cases hr : p.rms <;> simp [mean,hr,hi,Forward.binary,algebra,LayerNormLiger.Logical.algebra]

theorem variance_value (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) :
    Forward.varianceValue (algebra rest) p s = variance p.columns p.rms (input p s) := by
  change (∑ i : Fin p.block, (if i.val < p.columns then Forward.centered (algebra rest) p s (i,PUnit.unit) else 0) *
      (if i.val < p.columns then Forward.centered (algebra rest) p s (i,PUnit.unit) else 0)) / p.columns = _
  have he : (fun i : Fin p.block =>
      (if i.val < p.columns then Forward.centered (algebra rest) p s (i,PUnit.unit) else 0) *
      (if i.val < p.columns then Forward.centered (algebra rest) p s (i,PUnit.unit) else 0)) =
      fun i => if i.val < p.columns then (input p s i.val - mean p.columns p.rms (input p s))^2 else 0 := by
    funext i
    by_cases hi : i.val < p.columns
    · simp only [hi,if_true,centered_value rest p s hb (i,PUnit.unit) hi,pow_two]
    · simp [hi]
  rw [he,padded_sum _ _ hb (fun i => (input p s i - mean p.columns p.rms (input p s))^2)]
  rfl

theorem rstd_value (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) :
    Forward.rstdValue (algebra rest) p s = rstd p.columns p.rms p.epsilon (input p s) := by
  unfold Forward.rstdValue
  rw [variance_value rest p s hb]
  rfl

#axiomsClean z_value
#axiomsClean mean_value
#axiomsClean variance_value
#axiomsClean rstd_value
end
end VeriTile.Bench.Repairs.LayerNormOps.Logical
