import bench.repairs.tritonbench_g.spinning_lock_reduction.Kernels
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Host

/-! Address adapter for the separate i64 candidate. Unlike the original
HostBuffers interface, this interface contains no assumed equality between
fixed-width offsets/masks and mathematical coordinates. Those equalities are
derived below from ordinary allocation capacities and tensor metadata bounds.
P/C/locks remain separate caller-supplied windows; internal C overlap is allowed.
-/
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock

structure Buffers (h : HostArgs) where
  pElements : Nat
  cElements : Nat
  lockElements : Nat
  pBase : Nat
  cBase : Nat
  lockBase : Nat
  smsPositive : 0 < h.numSms
  smsGridLimit : h.numSms < 2 ^ 31
  groupPositive : 0 < h.groupK
  groupScalarLimit : h.groupK < 2 ^ 64
  tileMPositive : 0 < h.bm
  tileNPositive : 0 < h.bn
  tileMArange : h.bm ≤ 1048576
  tileNArange : h.bn ≤ 1048576
  mMetadata : h.m < 2 ^ 63
  nMetadata : h.n < 2 ^ 63
  strideMMetadata : h.strideM < 2 ^ 63
  strideNMetadata : h.strideN < 2 ^ 63
  partialCapacity : h.numSms * h.bm * h.bn ≤ pElements
  lockCapacity : h.numSms ≤ lockElements
  outputCapacity : ∀ row < h.m, ∀ col < h.n,
    row * h.strideM + col * h.strideN < cElements
  partialStorageBytes : 4 * pElements < 2 ^ 63
  outputStorageBytes : 4 * cElements < 2 ^ 63
  lockStorageBytes : 4 * lockElements < 2 ^ 63
  disjointPC : pBase + 4 * pElements ≤ cBase ∨ cBase + 4 * cElements ≤ pBase
  disjointPL : pBase + 4 * pElements ≤ lockBase ∨ lockBase + 4 * lockElements ≤ pBase
  disjointCL : cBase + 4 * cElements ≤ lockBase ∨ lockBase + 4 * lockElements ≤ cBase

theorem signed64_nat (n : Nat) (hn : n < 2 ^ 63) :
    (BitVec.ofNat 64 n).toInt = (n : Int) := by
  have hn64 : n < 18446744073709551616 := by omega
  have ht : (BitVec.ofNat 64 n).toNat = n := by simp [Nat.mod_eq_of_lt hn64]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [ht]; omega), ht]

theorem unsigned64_nat (n : Nat) (hn : n < 2 ^ 63) :
    (BitVec.ofNat 64 n).msb = false := by
  apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
  simp only [BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (show n < 2 ^ 64 by omega)]
  omega

theorem coordinates64 (h : HostArgs) (buf : Buffers h) (p lane : Nat) (hp : p < h.numSms) :
    row64 h p lane = BitVec.ofNat 64 (lane / h.bn) ∧
      col64 h p lane = BitVec.ofNat 64 (p * h.bn + lane % h.bn) := by
  have hn63 : h.numSms < 2 ^ 63 := by have := buf.smsGridLimit; omega
  have hp63 : p < 2 ^ 63 := Nat.lt_trans hp hn63
  have hn64 : h.numSms < 18446744073709551616 := by omega
  have hp64 : p < 18446744073709551616 := by omega
  have hd : (BitVec.ofNat 64 p).sdiv (BitVec.ofNat 64 h.numSms) = 0 := by
    apply BitVec.eq_of_toNat_eq
    simp [BitVec.sdiv_eq, unsigned64_nat p hp63, unsigned64_nat h.numSms hn63,
      Nat.mod_eq_of_lt hp64, Nat.mod_eq_of_lt hn64, Nat.div_eq_of_lt hp]
  have hm : (BitVec.ofNat 64 p).srem (BitVec.ofNat 64 h.numSms) = BitVec.ofNat 64 p := by
    apply BitVec.eq_of_toNat_eq
    simp [BitVec.srem_eq, unsigned64_nat p hp63, unsigned64_nat h.numSms hn63,
      Nat.mod_eq_of_lt hp64, Nat.mod_eq_of_lt hn64, Nat.mod_eq_of_lt hp]
  constructor
  · simp [row64, hd]
  · simp [col64, hm, BitVec.ofNat_mul, BitVec.ofNat_add]

theorem partial_nat_bounds (h : HostArgs) (buf : Buffers h) (q lane : Nat)
    (hq : q < h.numSms) (hl : lane < h.bm * h.bn) :
    q * h.bm * h.bn + lane < buf.pElements := by
  have hb := Nat.mul_le_mul_right (h.bm * h.bn) (show q + 1 ≤ h.numSms by omega)
  have cap := buf.partialCapacity
  simp only [Nat.add_mul, Nat.one_mul, Nat.mul_assoc] at *
  omega

theorem partial_offset64_exact (h : HostArgs) (buf : Buffers h) (q lane : Nat)
    (hq : q < h.numSms) (hl : lane < h.bm * h.bn) :
    partialOffset64 h q lane = (q * h.bm * h.bn + lane : Nat) := by
  have hb := partial_nat_bounds h buf q lane hq hl
  have hcap : buf.pElements < 2 ^ 63 := by have := buf.partialStorageBytes; omega
  have hdiv := Nat.div_mul_le_self lane h.bn
  have hmod := Nat.mod_le lane h.bn
  unfold partialOffset64
  simp only [← BitVec.ofNat_mul]
  rw [signed64_nat _ (by omega), signed64_nat _ (by omega), signed64_nat _ (by omega)]
  have he : lane / h.bn * h.bn + lane % h.bn = lane := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod lane h.bn
  rw [← Nat.cast_add, ← Nat.cast_add, Nat.add_assoc, he]

theorem coordinate_bounds (h : HostArgs) (buf : Buffers h) (p lane : Nat)
    (hp : p < h.numSms) (hl : lane < h.bm * h.bn) :
    lane / h.bn < 2 ^ 63 ∧ p * h.bn + lane % h.bn < 2 ^ 63 := by
  have hr : lane / h.bn < h.bm := (Nat.div_lt_iff_lt_mul buf.tileNPositive).mpr hl
  have hrem : lane % h.bn < h.bn := Nat.mod_lt _ buf.tileNPositive
  have hcol : p * h.bn + lane % h.bn < h.numSms * h.bn := by
    have hh := Nat.mul_le_mul_right h.bn (show p + 1 ≤ h.numSms by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hh
    omega
  have hn : h.numSms * h.bn ≤ h.numSms * h.bm * h.bn := by
    have hh := Nat.mul_le_mul_right h.numSms (show 1 ≤ h.bm from buf.tileMPositive)
    have := Nat.mul_le_mul_right h.bn hh
    simpa [Nat.mul_comm] using this
  have hcap := buf.partialCapacity
  have hbytes := buf.partialStorageBytes
  have hbm := buf.tileMArange
  constructor <;> omega

theorem mask64_exact (h : HostArgs) (buf : Buffers h) (p lane : Nat)
    (hp : p < h.numSms) (hl : lane < h.bm * h.bn) :
    outputMask64 h p lane = true ↔
      lane / h.bn < h.m ∧ p * h.bn + lane % h.bn < h.n := by
  obtain ⟨hr, hc⟩ := coordinates64 h buf p lane hp
  obtain ⟨hrb, hcb⟩ := coordinate_bounds h buf p lane hp hl
  simp only [outputMask64, hr, hc, Bool.and_eq_true, BitVec.slt_iff_toInt_lt]
  rw [signed64_nat _ hrb, signed64_nat _ hcb, signed64_nat _ buf.mMetadata,
    signed64_nat _ buf.nMetadata]
  omega

theorem output_offset64_exact (h : HostArgs) (buf : Buffers h) (p lane : Nat)
    (hp : p < h.numSms) (hl : lane < h.bm * h.bn)
    (hm : outputMask64 h p lane = true) :
    outputOffset64 h p lane =
      ((lane / h.bn) * h.strideM + (p * h.bn + lane % h.bn) * h.strideN : Nat) := by
  obtain ⟨hr, hc⟩ := coordinates64 h buf p lane hp
  obtain ⟨hrm, hcn⟩ := (mask64_exact h buf p lane hp hl).mp hm
  have hb := buf.outputCapacity _ hrm _ hcn
  have hbytes := buf.outputStorageBytes
  unfold outputOffset64
  rw [hr, hc]
  simp only [← BitVec.ofNat_mul]
  rw [signed64_nat _ (by omega), signed64_nat _ (by omega)]
  push_cast
  rfl

theorem output_offset64_in_allocation (h : HostArgs) (buf : Buffers h) (p lane : Nat)
    (hp : p < h.numSms) (hl : lane < h.bm * h.bn)
    (hm : outputMask64 h p lane = true) :
    0 ≤ outputOffset64 h p lane ∧ outputOffset64 h p lane < buf.cElements := by
  obtain ⟨hrm, hcn⟩ := (mask64_exact h buf p lane hp hl).mp hm
  rw [output_offset64_exact h buf p lane hp hl hm]
  exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (buf.outputCapacity _ hrm _ hcn)⟩

#axiomsClean partial_offset64_exact
#axiomsClean mask64_exact
#axiomsClean output_offset64_exact
#axiomsClean output_offset64_in_allocation
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide
