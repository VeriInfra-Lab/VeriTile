import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Host

/-! Concrete metadata from test_spinning_lock. The allocated buffers retain
304 tiles/locks while later calls use num_sms=2 and 5. No tensor/runtime
implementation is postulated; these are its caller-supplied buffer interfaces. -/
namespace VeriTile.Bench.TritonBenchG.SpinningLock

def fixtureArgs (sms k : Nat) : HostArgs := ⟨sms, k, 1024, 1024, 1024, 1, 128, 128⟩

theorem fixture_partial_offset (sms k q l : Nat) (hq : q < 304) (hl : l < 16384) :
    (fixtureArgs sms k).config.pAddr (BitVec.ofNat 32 q) l =
      (q * 128 * 128 + l : Nat) := by
  have hdiv : l / 128 < 128 := by omega
  have hmod : l % 128 < 128 := Nat.mod_lt _ (by decide)
  change (BitVec.ofNat 32 q * BitVec.ofNat 32 128 * BitVec.ofNat 32 128).toInt +
    (BitVec.ofNat 32 (l / 128) * BitVec.ofNat 32 128).toInt +
    (BitVec.ofNat 32 (l % 128)).toInt = _
  simp only [← BitVec.ofNat_mul]
  rw [signed32_nat _ (by omega), signed32_nat _ (by omega), signed32_nat _ (by omega)]
  have he := Nat.div_add_mod l 128
  push_cast
  omega

theorem fixture_row_offset (sms k p l : Nat) (hs : sms ≤ 304)
    (hp : p < sms) (hl : l < 16384) :
    ((fixtureArgs sms k).config.row p l).toInt = (l / 128 : Nat) := by
  rw [wrapper_row (fixtureArgs sms k) (by dsimp [fixtureArgs]; omega) p l hp]
  exact signed32_nat _ (by dsimp [fixtureArgs]; omega)

theorem fixture_col_offset (sms k p l : Nat) (hs : sms ≤ 304)
    (hp : p < sms) (_hl : l < 16384) :
    ((fixtureArgs sms k).config.col p l).toInt = (p * 128 + l % 128 : Nat) := by
  rw [wrapper_col (fixtureArgs sms k) (by dsimp [fixtureArgs]; omega) p l hp]
  apply signed32_nat
  dsimp [fixtureArgs]
  have hm := Nat.mod_lt l (show 0 < 128 by decide)
  omega

theorem fixture_output_offset (sms k p l : Nat) (hs : sms ≤ 304)
    (hp : p < sms) (hl : l < 16384) :
    (fixtureArgs sms k).config.cAddr p l = ((l / 128) * 1024 + (p * 128 + l % 128) : Nat) := by
  unfold Config.cAddr
  rw [wrapper_row (fixtureArgs sms k) (by dsimp [fixtureArgs]; omega) p l hp,
    wrapper_col (fixtureArgs sms k) (by dsimp [fixtureArgs]; omega) p l hp]
  change (BitVec.ofNat 32 (l / 128) * BitVec.ofNat 32 1024).toInt +
    (BitVec.ofNat 32 (p * 128 + l % 128) * BitVec.ofNat 32 1).toInt = _
  simp only [← BitVec.ofNat_mul]
  have hm := Nat.mod_lt l (show 0 < 128 by decide)
  rw [signed32_nat _ (by omega), signed32_nat _ (by omega)]
  simp

theorem fixture_mask (sms k p l : Nat) (hs : sms ≤ 304)
    (hp : p < sms) (hl : l < 16384) :
    (fixtureArgs sms k).config.outputMask p l = true ↔
      l / 128 < 1024 ∧ p * 128 + l % 128 < 1024 := by
  simp only [Config.outputMask, Bool.and_eq_true, BitVec.slt_iff_toInt_lt]
  rw [fixture_row_offset sms k p l hs hp hl, fixture_col_offset sms k p l hs hp hl]
  change (l / 128 : Int) < 1024 ∧ (p * 128 + l % 128 : Nat) < (1024 : Int) ↔ _
  omega

/-- All four source test calls share the original P/C/lock capacities. Arbitrary
fresh-lock memories can use this metadata, including nonzero initial P and C. -/
def fixtureBuffers (sms k : Nat) (hs0 : 0 < sms) (hs : sms ≤ 304)
    (hk0 : 0 < k) (hk : k ≤ 3) : HostBuffers (fixtureArgs sms k) where
  pElements := 304 * 128 * 128
  cElements := 1024 * 1024
  lockElements := 304
  pBase := 0
  cBase := 4 * (304 * 128 * 128)
  lockBase := 4 * (304 * 128 * 128) + 4 * (1024 * 1024)
  smsPositive := hs0
  smsSigned := by dsimp [fixtureArgs]; omega
  groupPositive := hk0
  groupSigned := by dsimp [fixtureArgs]; omega
  tileMPositive := by change 0 < 128; decide
  tileNPositive := by change 0 < 128; decide
  partialCapacity := by dsimp [fixtureArgs]; omega
  lockCapacity := hs
  outputCapacity := by
    intro r hr col hc
    dsimp [fixtureArgs] at *
    omega
  disjointPC := by decide
  disjointPL := by decide
  disjointCL := by decide
  partialOffset := by
    intro q hq l hl
    exact fixture_partial_offset sms k q l (by dsimp [fixtureArgs] at hq; omega) hl
  rowOffset := by
    intro p hp l hl
    exact fixture_row_offset sms k p l hs hp hl
  colOffset := by
    intro p hp l hl
    exact fixture_col_offset sms k p l hs hp hl
  outputOffset := by
    intro p hp l hl
    simpa [fixtureArgs] using fixture_output_offset sms k p l hs hp hl
  maskExact := by
    intro p hp l hl
    exact fixture_mask sms k p l hs hp hl

theorem original_four_calls_have_metadata :
    Nonempty (HostBuffers (fixtureArgs 304 3)) ∧
    Nonempty (HostBuffers (fixtureArgs 304 2)) ∧
    Nonempty (HostBuffers (fixtureArgs 2 2)) ∧
    Nonempty (HostBuffers (fixtureArgs 5 2)) :=
  ⟨⟨fixtureBuffers 304 3 (by decide) (by decide) (by decide) (by decide)⟩,
   ⟨fixtureBuffers 304 2 (by decide) (by decide) (by decide) (by decide)⟩,
   ⟨fixtureBuffers 2 2 (by decide) (by decide) (by decide) (by decide)⟩,
   ⟨fixtureBuffers 5 2 (by decide) (by decide) (by decide) (by decide)⟩⟩

#axiomsClean original_four_calls_have_metadata
end VeriTile.Bench.TritonBenchG.SpinningLock
