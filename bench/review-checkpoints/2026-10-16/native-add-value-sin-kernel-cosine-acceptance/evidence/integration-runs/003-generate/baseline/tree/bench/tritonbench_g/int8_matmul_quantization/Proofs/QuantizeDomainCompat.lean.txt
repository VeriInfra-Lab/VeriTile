import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeSource

namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Exact pre-migration comparison/select profile, retained only to audit the
accepted input domain. It is never substituted for the new source primitive. -/
def legacyPairMaximum (a b : Word) : Word := if FP.lt .ieeeValue b a then a else b

def legacyRunningMaximum (p : Parameters) (s : State Word) : Nat → Fin p.blockRows → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | k+1 => fun i => legacyPairMaximum (legacyRunningMaximum p s k i) (blockMaximum p s k i)

def legacyScale (cfg : FP.Config) (p : Parameters) (s : State Word) (i : Fin p.blockRows) : Word :=
  FP.div cfg (legacyRunningMaximum p s p.steps i) (FP.Value.ofBits .fp32 0x42fe0000)

def legacyQuotient (cfg : FP.Config) (p : Parameters) (s : State Word) (k : Nat) (i : Index p) : Word :=
  FP.div cfg (loaded p s k i) (legacyScale cfg p s i.1)

def LegacyInputDefined (cfg : FP.Config) (p : Parameters) (s : State Word) : Prop :=
  ∀ k, k < p.steps → ∀ i, active p k i → (toSigned 8 (legacyQuotient cfg p s k i)).isSome

private theorem div_nan_left (cfg : FP.Config) (x y : Word) (hx : x.isNaN = Bool.true) :
    FP.div cfg x y = FP.Value.qNaN .fp32 := by
  have hn : x.exponent = 255 ∧ x.fraction ≠ 0 := by simpa [FP.Value.isNaN] using hx
  simp [FP.div, FP.prepare, FP.Value.isSubnormal, FP.Value.decode, hn.1, hn.2]

private theorem defined_div_not_nan (cfg : FP.Config) (x y : Word)
    (hd : (toSigned 8 (FP.div cfg x y)).isSome) : x.isNaN = Bool.false := by
  cases hn : x.isNaN with
  | «false» => rfl
  | «true» =>
    rw [div_nan_left cfg x y hn] at hd
    have hbad : toSigned 8 (FP.Value.qNaN .fp32) = none := by decide +kernel
    simp [hbad] at hd

private theorem abs_nan_flag (x : Word) : x.abs.isNaN = x.isNaN := by
  have hm : x.raw % 2147483648 % 4294967296 = x.raw % 2147483648 :=
    Nat.mod_eq_of_lt (lt_trans (Nat.mod_lt x.raw (by decide : 0 < 2147483648)) (by decide))
  have he : x.abs.exponent = x.exponent := by
    change ((x.raw % 2147483648) % 4294967296 / 8388608) % 256 = (x.raw / 8388608) % 256
    rw [hm]
    rw [show 2147483648 = 8388608 * 256 from rfl, Nat.mod_mul_right_div_self, Nat.mod_mod]
  have hf : x.abs.fraction = x.fraction := by
    change ((x.raw % 2147483648) % 4294967296) % 8388608 = x.raw % 8388608
    rw [hm]
    exact Nat.mod_mod_of_dvd x.raw (by decide : 8388608 ∣ 2147483648)
  simp [FP.Value.isNaN, he, hf]

private theorem legacy_loaded_not_nan (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hd : LegacyInputDefined cfg p s) (k : Nat) (hk : k < p.steps) (i : Index p) :
    (loaded p s k i).isNaN = Bool.false := by
  by_cases ha : active p k i
  · exact defined_div_not_nan cfg _ _ (hd k hk i ha)
  · simp [loaded, ha]
    decide +kernel

private theorem legacy_block_not_nan (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hw : 0 < p.blockColumns) (hd : LegacyInputDefined cfg p s)
    (k : Nat) (hk : k < p.steps) (i : Fin p.blockRows) :
    (blockMaximum p s k i).isNaN = Bool.false := by
  have h : ∀ j : Fin p.blockColumns, ((loaded p s k (i,j,PUnit.unit)).abs).isNaN = Bool.false := by
    intro j
    rw [abs_nan_flag]
    exact legacy_loaded_not_nan cfg p s hd k hk _
  simpa [blockMaximum, QuantizeKvCopy.TypedQuantize.maximum,
    MatrixReduction.MachineMax.rowMaximum, hw] using
    h (MatrixReduction.MachineMax.best .fp32
      (fun j => ((loaded p s k (i,j,PUnit.unit)).abs).bits) hw)

/-- No extra non-NaN premise: it is derived from the old active conversion domain. -/
theorem legacy_running_compat (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hw : 0 < p.blockColumns) (hd : LegacyInputDefined cfg p s) :
    ∀ k, k ≤ p.steps → ∀ i,
      legacyRunningMaximum p s k i = runningMaximum p s k i ∧
      (legacyRunningMaximum p s k i).isNaN = Bool.false := by
  intro k
  induction k with
  | zero =>
    intro hk i
    constructor
    · rfl
    · change (FP.Value.zero .fp32).isNaN = Bool.false
      decide +kernel
  | succ k ih =>
    intro hk i
    obtain ⟨he, hn⟩ := ih (by omega) i
    have hb := legacy_block_not_nan cfg p s hw hd k (by omega) i
    constructor
    · simp [legacyRunningMaximum, runningMaximum, ← he, pairMaximum, legacyPairMaximum, hn, hb]
    · simp only [legacyRunningMaximum, legacyPairMaximum]
      split <;> assumption

theorem legacy_scale_compat (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hw : 0 < p.blockColumns) (hd : LegacyInputDefined cfg p s) (i : Fin p.blockRows) :
    legacyScale cfg p s i = scale cfg p s i := by
  simp only [legacyScale, scale, (legacy_running_compat cfg p s hw hd p.steps le_rfl i).1]

theorem legacy_quotient_compat (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hw : 0 < p.blockColumns) (hd : LegacyInputDefined cfg p s) (k : Nat) (i : Index p) :
    legacyQuotient cfg p s k i = quotient cfg p s k i := by
  simp only [legacyQuotient, quotient, legacy_scale_compat cfg p s hw hd]

/-- Every formerly accepted active conversion remains defined after correcting
maximum; widths, trip counts and the old domain are unchanged. -/
theorem legacy_input_domain_preserved (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hw : 0 < p.blockColumns) (hd : LegacyInputDefined cfg p s) : InputDefined cfg p s := by
  intro k hk i hi
  rw [← legacy_quotient_compat cfg p s hw hd]
  exact hd k hk i hi

#axiomsClean legacy_running_compat
#axiomsClean legacy_scale_compat
#axiomsClean legacy_quotient_compat
#axiomsClean legacy_input_domain_preserved
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
