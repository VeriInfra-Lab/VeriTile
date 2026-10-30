import bench.tritonbench_g.seeded_dropout.Proofs.WholeTile

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Input probability is mathematical. Its FP32 ABI conversion may round to
one, so this condition is never used to remove eager floating division. -/
def ProbabilityDomain (p : ℝ) : Prop := 0 ≤ p ∧ p < 1

/-- Independent mathematical formula, indexed by the original logical flat
position. The deterministic generator is not replaced by a statistical law. -/
def realFormula (p : ℝ) (seed : Int) (i : Nat) (x : ℝ) : ℝ :=
  if Random.randReal seed (i : Int) .bits32 10 > p then x/(1-p) else 0

/-- Pure Real interpretation, including the exact finite canonical random
payload decoder. This is not an IEEE backend. -/
def realAlgebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | _ => 0
  unary := fun _ _ x => x
  compareLt := fun _ _ => some (fun x y => decide (x < y))
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun word => (((FP.Value.mk word.bits : FP.Value .fp32).toRat?).getD 0 : ℝ)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} _ _ _ => fun _ => 0

theorem scalar_real (p : ℝ) (seed : Int) (i : Nat) (x : ℝ) :
    scalar realAlgebra (fun a b => decide (a < b)) p seed i x = realFormula p seed i x := by
  have random : ((((Random.rand seed (i : Int) .bits32 10).toRat?).getD 0 : Rat) : ℝ) =
      Random.randReal seed (i : Int) .bits32 10 := by
    rw [Random.rand_toRat_randRat]
    rfl
  simp only [scalar,realAlgebra,realFormula,id_eq,decide_eq_true_eq]
  change (if p < ((((Random.rand seed (i : Int) .bits32 10).toRat?).getD 0 : Rat) : ℝ)
    then x/(1-p) else 0.0) = _
  rw [random]
  norm_num

/-- Installed Python-int JIT binding: i32, then i64, then u64. The formal
kernel's additional u32 specialization is not a normal Python-int branch. -/
def PythonSeedABI (kind : Random.IntegerType) (seed : SeedCarrier kind) : Prop :=
  let value := seedInt kind seed
  (kind = .i32 ∧ -(2^31 : Int) ≤ value ∧ value < 2^31) ∨
  (kind = .i64 ∧ -(2^63 : Int) ≤ value ∧ value < 2^63 ∧
    (value < -(2^31 : Int) ∨ (2^31 : Int) ≤ value)) ∨
  (kind = .u64 ∧ (2^63 : Int) ≤ value ∧ value < 2^64)

theorem PythonSeedABI.valid (kind : Random.IntegerType) (seed : SeedCarrier kind)
    (abi : PythonSeedABI kind seed) : kind.inRange (seedInt kind seed) = Bool.true := by
  rcases abi with h | h | h
  · rcases h with ⟨hk,hlo,hhi⟩
    subst kind
    simpa only [Random.IntegerType.inRange,decide_eq_true_eq] using And.intro hlo hhi
  · rcases h with ⟨hk,hlo,hhi,_⟩
    subst kind
    simpa only [Random.IntegerType.inRange,decide_eq_true_eq] using And.intro hlo hhi
  · rcases h with ⟨hk,hlo,hhi⟩
    subst kind
    simp only [Random.IntegerType.inRange,decide_eq_true_eq]
    constructor <;> omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole
