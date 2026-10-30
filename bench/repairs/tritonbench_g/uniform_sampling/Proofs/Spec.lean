import bench.tritonbench_g.uniform_sampling.Proofs.ScalarSpec

/-! Input-only mathematical specification. No candidate execution, host action,
allocator, memory operation, or success condition occurs in the definitions. -/
namespace VeriTile.Bench.Repairs.UniformSampling.Spec
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section

def index (B i : Nat) : Nat := (i/(4*B))*B+i%B

def field (B i : Nat) : Random.Direct.Field :=
  match (i/B)%4 with |0=>.c0|1=>.c1|2=>.c2|_=>.c3

def counter (position : Nat) : Random.Counter :=
  ⟨BitVec.ofNat 32 position,BitVec.ofNat 32 (position/2^32),0,0⟩

def words (seed position : Nat) : Random.Counter :=
  Random.philox 10 (counter position) (Random.seedKey (seed : Int))

def word (seed start B i : Nat) : Random.Word :=
  ScalarSpec.field (words seed (start+index B i)) (field B i)

/-- Ideal real uniform quantile associated with the exact Philox integer. This
uses the source's sign-fold convention, before its two FP32 rounding steps. -/
def unit (w : Random.Word) : ℝ := (Random.foldSign w : ℝ)/(2:ℝ)^31

def target (lower upper : ℝ) (seed start B i : Nat) : ℝ :=
  lower+(upper-lower)*unit (word seed start B i)

/-- The canonical two-rounding decoder is also exposed as a separate, input-only
reference. This is a finite software distribution, not a statistical claim. -/
def canonicalTarget (lower upper : ℝ) (seed start B i : Nat) : ℝ :=
  lower+(upper-lower)*Random.uniformReal (word seed start B i)

theorem unit_range (w : Random.Word) : 0≤unit w ∧ unit w<1 := by
  have hlt : Random.foldSign w<2^31 := by
    unfold Random.foldSign
    split <;> have hw:=w.isLt <;> omega
  unfold unit
  constructor
  · positivity
  · apply (div_lt_one (by positivity : (0:ℝ)<2^31)).mpr
    exact_mod_cast hlt

theorem range (lower upper : ℝ) (seed start B i : Nat) (h:lower≤upper) :
    lower≤target lower upper seed start B i ∧ target lower upper seed start B i≤upper := by
  have hu:=unit_range (word seed start B i)
  unfold target
  constructor <;> nlinarith

#axiomsClean unit_range
#axiomsClean range
end
end VeriTile.Bench.Repairs.UniformSampling.Spec
