import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeScalarABI
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeHost

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
open VeriTile.Triton FP.Structural WholeSource
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open HostPlan SignedView SignedGrid
noncomputable section
local instance : Inhabited Float32Bits := ⟨⟨0⟩⟩

theorem fp32_scalar (rest : Algebra Float32Bits) (baseWord : Float32Bits)
    (base : ℝ) (exponent : Float32Bits) :
    WholeTile.scalarResult (fp32Backend rest baseWord) base exponent =
      rest.binary (some .fp32) .real .pow baseWord exponent := by
  rfl

/-- Concrete little-endian byte observation of an FP32 word. -/
def wordByte (word : Float32Bits) (byte : Nat) : Nat :=
  (word.bits.toNat / 2 ^ (8 * byte)) % 256

def memoryByte (s : State Float32Bits) (r : RegionName) (a byte : Nat) : Nat :=
  wordByte ((s.mem r a).read .fp32) byte

theorem output_bytes (s : State Float32Bits) (r : RegionName) (a : Nat)
    (word : Float32Bits) (cell : s.mem r a = .mk .fp32 word) (byte : Nat) :
    memoryByte s r a byte = wordByte word byte := by
  simp [memoryByte, cell]

theorem frame_bytes (s t : State Float32Bits) (r : RegionName) (a : Nat)
    (frame : t.mem r a = s.mem r a) (byte : Nat) :
    memoryByte t r a byte = memoryByte s r a byte := by
  simp [memoryByte, frame]

/-- Logical output equality plus the complete entry frame gives equality of
all raw memory, including holes of signed/noncontiguous output views. -/
theorem memory_equal_of_outputs_and_frame {α : Type} (input output : Buffer)
    (n : Nat) (sy : Int) (entry first second : State α)
    (sameOutputs : ∀ i, i < n → first.mem output.region ((output.view sy).address i) =
      second.mem output.region ((output.view sy).address i))
    (firstFrame : ∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
      first.mem r a = entry.mem r a)
    (secondFrame : ∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
      second.mem r a = entry.mem r a) : first.mem = second.mem := by
  classical
  funext r a
  by_cases hr : r = output.region
  · subst r
    by_cases active : ∃ i, i < n ∧ a = (output.view sy).address i
    · obtain ⟨i, hi, ha⟩ := active
      rw [ha]
      exact sameOutputs i hi
    · have outside : ∀ i, i < n → a ≠ (output.view sy).address i := by
        intro i hi ha
        exact active ⟨i, hi, ha⟩
      rw [firstFrame _ _ (Or.inr outside), secondFrame _ _ (Or.inr outside)]
  · rw [firstFrame _ _ (Or.inl hr), secondFrame _ _ (Or.inl hr)]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
