import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_ZeroInvariant
import bench.tritonbench_g.spinning_lock_reduction.Sources.Mathematical_HostSource

namespace VeriTile.Bench.TritonBenchG.SpinningLock

theorem wrapper_grid (h : HostArgs) : h.config.grid = h.numSms := rfl
theorem wrapper_returns_none : wrapperReturn = none := rfl

theorem signed32_nat (n : Nat) (hn : n < 2 ^ 31) :
    (BitVec.ofNat 32 n).toInt = (n : Int) := by
  have hn32 : n < 4294967296 := by omega
  have ht : (BitVec.ofNat 32 n).toNat = n := by simp [Nat.mod_eq_of_lt hn32]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [ht]; omega), ht]

theorem unsigned32_nat (n : Nat) (hn : n < 2 ^ 31) :
    (BitVec.ofNat 32 n).msb = false := by
  apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
  simp only [BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (show n < 2 ^ 32 by omega)]
  omega

/-- Because the exact wrapper grid is num_sms, every active pid_m is zero.
Thus this launch covers only the first BLOCK_SIZE_M rows. -/
theorem wrapper_program_coordinates (h : HostArgs) (hsm : h.numSms < 2 ^ 31)
    (p : Nat) (hp : p < h.config.grid) :
    (h.config.pid p).sdiv h.config.numSms = 0 ∧
      (h.config.pid p).srem h.config.numSms = h.config.pid p := by
  have hp' : p < h.numSms := hp
  have hp31 : p < 2 ^ 31 := Nat.lt_trans hp' hsm
  have hp32 : p < 4294967296 := by omega
  have hn32 : h.numSms < 4294967296 := by omega
  have hpmsb := unsigned32_nat p hp31
  have hnmsb := unsigned32_nat h.numSms hsm
  constructor
  · apply BitVec.eq_of_toNat_eq
    simp [HostArgs.config, Config.pid, BitVec.sdiv_eq, hpmsb, hnmsb,
      Nat.mod_eq_of_lt hp32, Nat.mod_eq_of_lt hn32, Nat.div_eq_of_lt hp']
  · apply BitVec.eq_of_toNat_eq
    simp [HostArgs.config, Config.pid, BitVec.srem_eq, hpmsb, hnmsb,
      Nat.mod_eq_of_lt hp32, Nat.mod_eq_of_lt hn32, Nat.mod_eq_of_lt hp']

theorem wrapper_row (h : HostArgs) (hsm : h.numSms < 2 ^ 31)
    (p lane : Nat) (hp : p < h.config.grid) :
    h.config.row p lane = BitVec.ofNat 32 (lane / h.bn) := by
  simp only [Config.row, (wrapper_program_coordinates h hsm p hp).1]
  simp [HostArgs.config]

theorem wrapper_col (h : HostArgs) (hsm : h.numSms < 2 ^ 31)
    (p lane : Nat) (hp : p < h.config.grid) :
    h.config.col p lane = BitVec.ofNat 32 (p * h.bn + lane % h.bn) := by
  simp only [Config.col, (wrapper_program_coordinates h hsm p hp).2]
  simp [HostArgs.config, Config.pid, BitVec.ofNat_add, BitVec.ofNat_mul]

/-- Explicit host interface conditions: allocations are supplied by the caller,
all signed-i32 address calculations agree with these integer offsets, and
separate regions correspond to distinct P/C/locks allocations. Address equations
are arithmetic metadata conditions, never assumptions on an execution result. -/
structure HostBuffers (h : HostArgs) where
  pElements : Nat
  cElements : Nat
  lockElements : Nat
  pBase : Nat
  cBase : Nat
  lockBase : Nat
  smsPositive : 0 < h.numSms
  smsSigned : h.numSms < 2 ^ 31
  groupPositive : 0 < h.groupK
  groupSigned : h.groupK < 2 ^ 31
  tileMPositive : 0 < h.bm
  tileNPositive : 0 < h.bn
  partialCapacity : h.numSms * h.bm * h.bn ≤ pElements
  lockCapacity : h.numSms ≤ lockElements
  outputCapacity : ∀ r < h.m, ∀ col < h.n,
    r * h.strideM + col * h.strideN < cElements
  disjointPC : pBase + 4 * pElements ≤ cBase ∨ cBase + 4 * cElements ≤ pBase
  disjointPL : pBase + 4 * pElements ≤ lockBase ∨ lockBase + 4 * lockElements ≤ pBase
  disjointCL : cBase + 4 * cElements ≤ lockBase ∨ lockBase + 4 * lockElements ≤ cBase
  partialOffset : ∀ q < h.numSms, ∀ l < h.config.lanes,
    h.config.pAddr (BitVec.ofNat 32 q) l = (q * h.bm * h.bn + l : Nat)
  rowOffset : ∀ p < h.numSms, ∀ l < h.config.lanes,
    (h.config.row p l).toInt = (l / h.bn : Nat)
  colOffset : ∀ p < h.numSms, ∀ l < h.config.lanes,
    (h.config.col p l).toInt = (p * h.bn + l % h.bn : Nat)
  outputOffset : ∀ p < h.numSms, ∀ l < h.config.lanes,
    h.config.cAddr p l = ((l / h.bn) * h.strideM + (p * h.bn + l % h.bn) * h.strideN : Nat)
  maskExact : ∀ p < h.numSms, ∀ l < h.config.lanes,
    h.config.outputMask p l = true ↔ l / h.bn < h.m ∧ p * h.bn + l % h.bn < h.n

/-- Static footprint of all active output lanes in the exact host grid. -/
def HostArgs.outputFootprint (h : HostArgs) (a : Int) : Prop :=
  ∃ p < h.numSms, ∃ l < h.config.lanes,
    h.config.outputMask p l = true ∧ h.config.cAddr p l = a

theorem output_offset_in_allocation (h : HostArgs) (buffers : HostBuffers h)
    (p l : Nat) (hp : p < h.numSms) (hl : l < h.config.lanes)
    (hm : h.config.outputMask p l = true) :
    0 ≤ h.config.cAddr p l ∧ h.config.cAddr p l < buffers.cElements := by
  rcases (buffers.maskExact p hp l hl).mp hm with ⟨hr, hc⟩
  rw [buffers.outputOffset p hp l hl]
  exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (buffers.outputCapacity _ hr _ hc)⟩

theorem partial_offset_in_allocation (h : HostArgs) (buffers : HostBuffers h)
    (q l : Nat) (hq : q < h.numSms) (hl : l < h.config.lanes) :
    0 ≤ h.config.pAddr (BitVec.ofNat 32 q) l ∧
      h.config.pAddr (BitVec.ofNat 32 q) l < buffers.pElements := by
  rw [buffers.partialOffset q hq l hl]
  constructor
  · exact Int.natCast_nonneg _
  · apply Int.ofNat_lt.mpr
    have hb : l < h.bm * h.bn := hl
    have ht : (q + 1) * (h.bm * h.bn) ≤ h.numSms * (h.bm * h.bn) :=
      Nat.mul_le_mul_right _ (by omega)
    have cap := buffers.partialCapacity
    simp only [Nat.add_mul, Nat.one_mul, Nat.mul_assoc] at *
    omega

#axiomsClean wrapper_program_coordinates
#axiomsClean partial_offset_in_allocation
#axiomsClean output_offset_in_allocation
end VeriTile.Bench.TritonBenchG.SpinningLock
