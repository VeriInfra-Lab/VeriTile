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

theorem fp32_complete (rest : Algebra Float32Bits) (baseWord : Float32Bits) (base : ℝ) (candidate : Bool) (input output : Buffer)
    {n inputCapacity outputCapacity : Nat} (positive : 0 < n) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (s : State Float32Bits) (xs : Nat → Float32Bits)
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    ∃ final, runHost (fp32Backend rest baseWord) base candidate input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (rest.binary (some .fp32) .real .pow baseWord (xs i))) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) := by
  obtain ⟨final, run, outputValues, inputValues, frame, bounds⟩ :=
    WholeHost.rank_one_success (fp32Backend rest baseWord) base candidate input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
      hin hout hstride hsep s
  refine ⟨final, run, ?_, inputValues, frame, bounds⟩
  intro i hi
  rw [outputValues i hi, values i hi]
  simp [Cell.read, fp32_scalar]

theorem python_fp32_complete (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat) (candidate : Bool) (input output : Buffer)
    {n inputCapacity outputCapacity : Nat} (positive : 0 < n) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (s : State Float32Bits) (xs : Nat → Float32Bits)
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    ∃ final, runPythonHost rest abi host candidate input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (rest.binary (some .fp32) .real .pow (abi.extractFP32 host.payload) (xs i))) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) := by
  rw [python_run_eq]
  exact fp32_complete rest (abi.extractFP32 host.payload) sourceSlot candidate input output positive sx sy
    hshapeIn hshapeOut hstrideIn hstrideOut hin hout hstride hsep s xs values

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
