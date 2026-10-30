import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardModel
import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardModel
namespace VeriTile.Bench.Repairs.LayerNormLiger.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive)
noncomputable section
open scoped BigOperators

def sum (n : Nat) (f : Nat → ℝ) := ∑ i ∈ Finset.range n, f i
/-- Population statistics and affine normalization over logical columns only. -/
def mean (n : Nat) (x : Nat → ℝ) := sum n x / n
def variance (n : Nat) (x : Nat → ℝ) := sum n (fun i => (x i - mean n x)^2) / n
def rstd (n : Nat) (eps : ℝ) (x : Nat → ℝ) := (Real.sqrt (variance n x + eps))⁻¹
def y (n : Nat) (eps : ℝ) (x w b : Nat → ℝ) (i : Nat) :=
  (x i-mean n x)*rstd n eps x*w i+b i
/-- Backward target uses the forward's saved population statistics. -/
def xhat (x : Nat → ℝ) (mu inv : ℝ) (i : Nat) := (x i-mu)*inv
def dx (n : Nat) (x w dy : Nat → ℝ) (mu inv : ℝ) (i : Nat) :=
  (w i*dy i-(xhat x mu inv i*(sum n (fun j => xhat x mu inv j*(w j*dy j))/n)
    +sum n (fun j => w j*dy j)/n))*inv
def dw (rows : Nat) (x dy : Nat → Nat → ℝ) (mu inv : Nat → ℝ) (i : Nat) :=
  sum rows (fun row => dy row i*xhat (x row) (mu row) (inv row) i)
def db (rows : Nat) (dy : Nat → Nat → ℝ) (i : Nat) := sum rows (fun row => dy row i)

def sumValues {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ∑ k : Fin (TileShape.axisDim shape ax), xs (TileShape.insertAxisIndex shape ax out k)
  · exact fun out => ∑ k : Fin (TileShape.axisDim shape ax), xs (TileShape.replaceAxisIndex shape ax out k)

def sumPrimitive : SumPrimitive ℝ where
  apply := fun _ {_} ax keep xs => sumValues ax keep xs

/-- Ideal real interpretation of the scalar leaves. Reductions are expanded
finite sums; no per-kernel output law is an assumption of this interpreter. -/
def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    cast := fun _ _ _ x => x
    binary := fun _ _ op x y => match op with
      | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
      | .max => max x y | .pow => Real.rpow x y
    unary := fun p op x => match op with
      | .rsqrt => (Real.sqrt x)⁻¹
      | _ => rest.unary p op x
    reduceSum := fun _ {_} ax keep xs => sumValues ax keep xs }

theorem sumValues_vector (n : Nat) (f : TileIndex [n] → ℝ) :
    sumValues ⟨0,by simp⟩ Bool.false f PUnit.unit = ∑ i : Fin n, f (i,PUnit.unit) := by rfl

theorem padded_sum (n block : Nat) (h : n ≤ block) (f : Nat → ℝ) :
    (∑ i : Fin block, if i.val<n then f i.val else 0) = sum n f := by
  rw [Fin.sum_univ_eq_sum_range (fun i => if i<n then f i else 0)]
  unfold sum
  calc
    (∑ i ∈ Finset.range block, if i<n then f i else 0) =
      ∑ i ∈ Finset.range n, if i<n then f i else 0 := by
        symm
        apply Finset.sum_subset (Finset.range_mono h)
        intro i hi hni
        simp only [Finset.mem_range, not_lt] at hni
        simp [Nat.not_lt.mpr hni]
    _ = ∑ i ∈ Finset.range n, f i := by
      apply Finset.sum_congr rfl
      intro i hi
      simp [Finset.mem_range.mp hi]

#axiomsClean sumValues_vector
#axiomsClean padded_sum
end
end VeriTile.Bench.Repairs.LayerNormLiger.Logical
