import bench.tritonbench_g.relu_triton_kernel.Proofs.Mathematical_TypedGrid
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.SoftwareFP32
open VeriTile.Triton FP.Structural GatedGrid TypedValue
noncomputable section
set_option maxHeartbeats 3000000

local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Explicit IEEE bit-order comparison: unordered NaNs are false, both zero
signs compare equal, and negative finite/infinite magnitudes reverse order.
No DAZ/FTZ preprocessing is introduced by this software profile. -/
def orderedLe (x y : FP.Value .fp32) : Bool :=
  if x.isNaN || y.isNaN then Bool.false
  else if x.isZero && y.isZero then Bool.true
  else if x.sign != y.sign then x.sign
  else if x.sign then decide (y.bits.toNat ≤ x.bits.toNat)
  else decide (x.bits.toNat ≤ y.bits.toNat)

/-- Ordered compare plus select, preserving a selected -0 word. This is not
an algebraic max rewrite, and does not claim native CUDA conformance. -/
def wordRelu (x : FP.Value .fp32) : FP.Value .fp32 :=
  if orderedLe (FP.Value.zero .fp32) x then x else FP.Value.zero .fp32

def expected (x y : RegionName) (N B : Nat)
    (mem : Memory (FP.Value .fp32)) : Memory (FP.Value .fp32) :=
  fun r o => if gridCount N B ≠ 0 ∧ r = y ∧ o < B ∧ o < N then
    .mk .fp32 (wordRelu ((mem x o).read .fp32)) else mem r o

theorem exact_grid (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (cmp16 : FP.Value .fp32 → FP.Value .fp32 → Bool)
    (x y : RegionName) (N B : Nat) (initial : State (FP.Value .fp32)) :
    ∃ final, runGrid (backend (FP.Software.scalarFP32 rest cfg) cmp16 (orderedLe))
        (source .fp32 x y N B) N B initial = some final ∧
      final.mem = expected x y N B initial.mem := by
  obtain ⟨final, hf, hm⟩ := grid_correct (FP.Software.scalarFP32 rest cfg)
    cmp16 (orderedLe) .fp32 x y N B initial
  refine ⟨final, hf, ?_⟩
  rw [hm]
  funext r o
  by_cases h : gridCount N B = 0
  · simp [h, expected]
  · simp [h, expected, reference, resultValue, Profile.kind, wordRelu,
      FP.Software.scalarFP32, FP.Software.fp32Context]

theorem quiet_nan_becomes_positive_zero :
    (wordRelu (FP.Value.ofBits .fp32 0x7fc12345)).bits = 0 := by decide

theorem signaling_nan_becomes_positive_zero :
    (wordRelu (FP.Value.ofBits .fp32 0x7f800001)).bits = 0 := by decide

theorem negative_zero_is_preserved :
    (wordRelu (FP.Value.ofBits .fp32 0x80000000)).bits = 0x80000000 := by decide

theorem positive_infinity_is_preserved :
    (wordRelu (FP.Value.ofBits .fp32 0x7f800000)).bits = 0x7f800000 := by decide

theorem negative_infinity_becomes_positive_zero :
    (wordRelu (FP.Value.ofBits .fp32 0xff800000)).bits = 0 := by decide

def harnessInputWords : List (List Nat) := [
  [0xc0400000, 0xbf800000, 0xbf000000, 0xc0000000, 0xc0a00000],
  [0x40400000, 0x3f800000, 0x3f000000, 0x40000000, 0x40a00000],
  [0xc0400000, 0xbf800000, 0x00000000, 0x40000000, 0x40a00000],
  [0, 0, 0, 0, 0] ]

def harnessExpectedWords : List (List Nat) := [
  [0, 0, 0, 0, 0],
  [0x40400000, 0x3f800000, 0x3f000000, 0x40000000, 0x40a00000],
  [0, 0, 0, 0x40000000, 0x40a00000],
  [0, 0, 0, 0, 0] ]

/-- Exact software scalar outputs for the four declared harness value lists.
Native Torch literal construction, allocation and harness execution remain
separate from this word-level calculation. -/
theorem harness_scalar_words :
    harnessInputWords.map (List.map (fun word =>
      (wordRelu (FP.Value.ofBits .fp32 word)).bits.toNat)) = harnessExpectedWords := by
  decide

#specNonCircular expected avoiding [source, materialized, runPrefix, runGrid, FP.Structural.exec]
#axiomsClean exact_grid
#axiomsClean quiet_nan_becomes_positive_zero
#axiomsClean signaling_nan_becomes_positive_zero
#axiomsClean negative_zero_is_preserved
#axiomsClean positive_infinity_is_preserved
#axiomsClean negative_infinity_becomes_positive_zero
#axiomsClean harness_scalar_words
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.SoftwareFP32
