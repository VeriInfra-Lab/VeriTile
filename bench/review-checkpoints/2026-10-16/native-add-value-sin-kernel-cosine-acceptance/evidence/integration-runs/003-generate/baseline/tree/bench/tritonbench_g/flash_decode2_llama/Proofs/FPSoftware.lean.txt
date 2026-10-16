import bench.tritonbench_g.flash_decode2_llama.Proofs.FPContract

/-! Concrete software binary32 division/copy and maximum, independent of any
opaque backend callback. The default maxnum profile selects the numeric operand for a single NaN and
selects a canonical NaN for two NaNs and +0 for a mixed-sign zero tie; those are explicit choices awaiting
native binding, not consequences of a comparison/select identity.
-/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := FP.Value .fp32

def roundedMaximum (cfg : FP.Config) (x y : Word) : Word :=
  let x := FP.prepare cfg x
  let y := FP.prepare cfg y
  if x.isNaN then (if y.isNaN then FP.Value.qNaN .fp32 else y)
  else if y.isNaN then x
  else if x.isZero && y.isZero then FP.Value.zero .fp32 (x.sign && y.sign)
  else if FP.lt cfg y x then x else y

/-- Only fp32 maximum is added to the scalar software adapter. All operations
used by the three stage results are discharged below without using `rest`. -/
def bitBackend (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest cfg
  { base with binary := fun p d op x y =>
      if FP.Software.fp32Context p d && decide (op = .max) then roundedMaximum cfg x y
      else base.binary p d op x y }

def roundedNormalization [Inhabited Word] (cfg : FP.Config) (s : State Word)
    (Acc SumExp : RegionName) (a : Row) (z : Scalar) (D : Nat) : TileIndex [D] → Word :=
  fun i => FP.div cfg ((s.mem Acc (address s a i)).read .real)
    ((s.mem SumExp (scalarAddress s z)).read .real)

def copiedBits [Inhabited Word] (s : State Word) (Final : RegionName)
    (a : Row) (D : Nat) : TileIndex [D] → Word :=
  fun i => (s.mem Final (address s a i)).read .real

def maximumBits [Inhabited Word] (cfg : FP.Config) (s : State Word)
    (LogExp Prev : RegionName) (block : Nat) (l m : Scalar) : Word :=
  roundedMaximum cfg ((s.mem LogExp (scalarAddress s l + block)).read .real)
    ((s.mem Prev (scalarAddress s m)).read .real)

theorem normalization_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (s : State Word) (Acc SumExp : RegionName) (a : Row) (z : Scalar) (D : Nat) :
    normalizationValues (bitBackend rest cfg) s Acc SumExp a z D =
      roundedNormalization cfg s Acc SumExp a z D := by
  funext i
  simp [normalizationValues, roundedNormalization, bitBackend, FP.Software.scalarFP32,
    FP.Software.fp32Context]

theorem copy_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (s : State Word) (Final : RegionName) (a : Row) (D : Nat) :
    copyValues (bitBackend rest cfg) s Final a D = copiedBits s Final a D := rfl

theorem maximum_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (s : State Word) (LogExp Prev : RegionName) (block : Nat) (l m : Scalar) :
    maximumValue (bitBackend rest cfg) s LogExp Prev block l m =
      maximumBits cfg s LogExp Prev block l m := by
  simp [maximumValue, maximumBits, bitBackend, FP.Software.scalarFP32,
    FP.Software.fp32Context]

/-- These include a rounded division and the zero/NaN cases that real arithmetic
or a comparison/select maximum would erase. Checked by Lean's kernel. -/
theorem scalar_boundary_vectors :
    (FP.div .ieeeValue (FP.Value.ofBits .fp32 0x3f800000)
      (FP.Value.ofBits .fp32 0x40400000)).bits = BitVec.ofNat 32 0x3eaaaaab ∧
    (FP.div .ieeeValue (FP.Value.zero .fp32) (FP.Value.zero .fp32)).isNaN = Bool.true ∧
    (FP.div .ieeeValue (FP.Value.zero .fp32 Bool.true)
      (FP.Value.ofBits .fp32 0x3f800000)).bits = BitVec.ofNat 32 0x80000000 ∧
    (roundedMaximum .ieeeValue (FP.Value.zero .fp32)
      (FP.Value.zero .fp32 Bool.true)).bits = BitVec.ofNat 32 0 ∧
    (roundedMaximum .ieeeValue (FP.Value.qNaN .fp32)
      (FP.Value.ofBits .fp32 0x3f800000)).bits = BitVec.ofNat 32 0x3f800000 := by
  decide +kernel

/-- Regression against the former frontend lowering: ordered comparison
followed by selection wrongly selects a NaN in the second operand. -/
theorem compare_select_is_not_maximum :
    (let x := FP.Value.ofBits .fp32 0x3f800000
     let y := FP.Value.qNaN .fp32
     if FP.lt .ieeeValue y x then x else y) ≠
      roundedMaximum .ieeeValue (FP.Value.ofBits .fp32 0x3f800000)
        (FP.Value.qNaN .fp32) := by
  decide +kernel

def maximumWitness [Inhabited Word] (x y : Word) : State Word where
  mem := fun r o => .mk .real (if r = "logic" ∧ o = 65 then x
    else if r = "previous" ∧ o = 31 then y else FP.Value.ofBits .fp32 0x12345678)
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 2 else 3
  numPids := fun _ => 4
  undef := fun dtype _ _ => defaultValue dtype

/-- Successful execution of the original maximum source at nonzero program
coordinates and distinct input offsets, for arbitrary input bit patterns. -/
theorem maximum_bit_execution [Inhabited Word] (rest : Algebra Word) (x y : Word) :
    ∃ final, FP.Structural.exec (bitBackend rest .ieeeValue)
      (maximum "logic" "previous" "next" 4 ⟨11,13⟩ ⟨5,7⟩) (maximumWitness x y) = some final ∧
      final.mem "next" 31 = .mk .real (roundedMaximum .ieeeValue x y) ∧
      final.mem "next" 32 = (maximumWitness x y).mem "next" 32 ∧
      final.mem "logic" 65 = (maximumWitness x y).mem "logic" 65 := by
  obtain ⟨final, he, hm⟩ := maximum_memory (bitBackend rest .ieeeValue)
    "logic" "previous" "next" 4 ⟨11,13⟩ ⟨5,7⟩ (maximumWitness x y)
  have hv : maximumValue (bitBackend rest .ieeeValue) (maximumWitness x y)
      "logic" "previous" 4 ⟨11,13⟩ ⟨5,7⟩ = roundedMaximum .ieeeValue x y := by
    rw [maximum_software]
    simp [maximumBits, maximumWitness, scalarAddress]
  refine ⟨final, he, ?_, ?_, ?_⟩
  all_goals rw [hm, hv]
  all_goals simp [State.write, scalarAddress, maximumWitness]

theorem maximum_nan_bit_success [Inhabited Word] (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (bitBackend rest .ieeeValue)
      (maximum "logic" "previous" "next" 4 ⟨11,13⟩ ⟨5,7⟩)
      (maximumWitness (FP.Value.ofBits .fp32 0x3f800000) (FP.Value.qNaN .fp32)) = some final ∧
      final.mem "next" 31 = .mk .real (FP.Value.ofBits .fp32 0x3f800000) ∧
      final.mem "next" 32 = .mk .real (FP.Value.ofBits .fp32 0x12345678) := by
  obtain ⟨final, he, hv, hf, _⟩ := maximum_bit_execution rest
    (FP.Value.ofBits .fp32 0x3f800000) (FP.Value.qNaN .fp32)
  have hn : roundedMaximum .ieeeValue (FP.Value.ofBits .fp32 0x3f800000)
      (FP.Value.qNaN .fp32) = FP.Value.ofBits .fp32 0x3f800000 := by decide +kernel
  exact ⟨final, he, hn ▸ hv, by simpa [maximumWitness] using hf⟩

#axiomsClean maximum_bit_execution
#axiomsClean maximum_nan_bit_success

#axiomsClean normalization_software
#axiomsClean copy_software
#axiomsClean maximum_software
#axiomsClean scalar_boundary_vectors
#axiomsClean compare_select_is_not_maximum
end
end VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages
