import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Numerics
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeAccess
import bench.tritonbench_g.pow_scalar_tensor.Kernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerKernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerPrecision
import bench.tritonbench_g.pow_scalar_tensor.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv
open VeriTile.Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) : Program where
  kernel := TypedSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B
  domain := []

/-- Full typed state/failure equivalence for the complete fp32-exponent source,
both constexpr branches and every formal floating destination. Only the
natural tile-offset product changes, so no numerical FP assumptions apply.
Signed strides and the separate native int32-exponent path are not covered. -/
specification floating_source_equivalence (output : TypedSource.FloatingOutput)
    (oneTile : Bool) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    program output oneTile Bool.false base X Y sx sy N tasks tiles B ≡[assumptions]
      program output oneTile Bool.true base X Y sx sy N tasks tiles B := by
  cases output <;> cases oneTile <;> equiv_decompose <;> fp_prove

def integerProgram (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) : Program where
  kernel := TypedIntegerSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B
  domain := []

/-- Complete state/failure equality for the native int32 exponent source and
all six floating destinations. The original integer values, range checks and
native powi callback are retained, including missing-backend failure. Only the
natural offset product changes; no numerical FP assumption is required. -/
specification integer_source_equivalence (output : TypedSource.FloatingOutput)
    (oneTile : Bool) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    integerProgram output oneTile Bool.false base X Y sx sy N tasks tiles B ≡[assumptions]
      integerProgram output oneTile Bool.true base X Y sx sy N tasks tiles B := by
  cases output <;> cases oneTile <;> equiv_decompose <;> fp_prove

#print_fp_assumptions floating_source_equivalence
#print_fp_assumptions integer_source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
open VeriTile.Triton FP.Structural WholeSource
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open HostPlan SignedView SignedGrid
noncomputable section
local instance : Inhabited Float32Bits := ⟨⟨0⟩⟩

/-- Complete FP32 source outcome, with every exponent word and every Python-float host payload
admitted through the explicit scalar extraction primitive. The unchanged libdevice power primitive remains opaque; no real-power,
finite-value or NaN-elimination law is assumed. Both source variants retain the
same scalar sequence. Addresses, returned object, all output bytes and the
entry-relative frame are derived from the complete host/grid execution. -/
specification pow_scalar_tensor_fp32_source_outcomes
    (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat) (candidate : Bool)
    (input output : Buffer) {n inputCapacity outputCapacity device : Nat}
    (positive : 0 < n) (bounded : n < SignedIndex32.indexLimit) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (_deviceIn : input.deviceIndex = some device) (_deviceOut : output.deviceIndex = some device)
    (_inputStrideWord : -(2^63 : Int) ≤ sx ∧ sx < 2^63)
    (_outputStrideWord : -(2^63 : Int) ≤ sy ∧ sy < 2^63)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (buffers : NativeBuffers input output inputCapacity outputCapacity)
    (s : State Float32Bits) (xs : Nat → Float32Bits)
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    ∃ final,
      runPythonHost rest abi host candidate input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (rest.binary (some .fp32) .real .pow (abi.extractFP32 host.payload) (xs i))) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) ∧
      NativeAccess input output n sx sy buffers ∧
      SourceAddressLowering input output n sx sy buffers ∧
      (∀ i j u v, i < n → j < n → u < 4 → v < 4 →
        buffers.inputBase + 4 * (input.view sx).address i + u ≠
          buffers.outputBase + 4 * (output.view sy).address j + v) ∧
      (∀ i, i < n → ∀ byte, byte < 4 →
        memoryByte final output.region ((output.view sy).address i) byte =
          wordByte (rest.binary (some .fp32) .real .pow (abi.extractFP32 host.payload) (xs i)) byte) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        ∀ byte, byte < 4 → memoryByte final r a byte = memoryByte s r a byte) := by
  obtain ⟨final, run, outputValues, inputValues, frame, bounds⟩ :=
    python_fp32_complete rest abi host candidate input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
      hin hout hstride hsep s xs values
  refine ⟨final, run, outputValues, inputValues, frame, bounds,
    native_access input output positive bounded sx sy hin hout buffers,
    source_address_lowering input output positive bounded sx sy hin hout buffers, ?_, ?_, ?_⟩
  · intro i j u v hi hj hu hv
    exact input_output_bytes_disjoint input output sx sy hin hout hsep buffers hi hj hu hv
  · intro i hi byte _
    exact output_bytes final output.region ((output.view sy).address i) _ (outputValues i hi) byte
  · intro r a outside byte _
    exact frame_bytes s final r a (frame r a outside) byte

/-- Full returned-memory equivalence of the actual original/candidate host
executions, including layout holes and all other allocations. Only integer
tile-offset multiplication changes. No numerical power law is needed. -/
specification pow_scalar_tensor_fp32_host_equivalence
    (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat)
    (input output : Buffer) {n inputCapacity outputCapacity device : Nat}
    (positive : 0 < n) (bounded : n < SignedIndex32.indexLimit) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (_deviceIn : input.deviceIndex = some device) (_deviceOut : output.deviceIndex = some device)
    (_inputStrideWord : -(2^63 : Int) ≤ sx ∧ sx < 2^63)
    (_outputStrideWord : -(2^63 : Int) ≤ sy ∧ sy < 2^63)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (buffers : NativeBuffers input output inputCapacity outputCapacity)
    (s : State Float32Bits) (xs : Nat → Float32Bits)
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    ∃ original candidate,
      runPythonHost rest abi host Bool.false input output Bool.true s = .returned output original ∧
      runPythonHost rest abi host Bool.true input output Bool.true s = .returned output candidate ∧
      original.mem = candidate.mem ∧ NativeAccess input output n sx sy buffers := by
  obtain ⟨original, ho, vo, _, fo, _⟩ :=
    python_fp32_complete rest abi host Bool.false input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
      hin hout hstride hsep s xs values
  obtain ⟨candidate, hc, vc, _, fc, _⟩ :=
    python_fp32_complete rest abi host Bool.true input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
      hin hout hstride hsep s xs values
  refine ⟨original, candidate, ho, hc, ?_, native_access input output positive bounded sx sy hin hout buffers⟩
  apply memory_equal_of_outputs_and_frame input output n sy s original candidate _ fo fc
  intro i hi
  rw [vo i hi, vc i hi]

theorem pow_scalar_tensor_host_failure (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat)
     (candidate : Bool) (input output : Buffer) (guardSucceeds : Bool)
    (s : State Float32Bits) (failure : Failure)
    (failed : prepare input output guardSucceeds = .error failure) :
    runPythonHost rest abi host candidate input output guardSucceeds s =
      .hostFailure failure s :=
  python_host_failure rest abi host candidate input output guardSucceeds s failure failed

#axiomsClean pow_scalar_tensor_fp32_source_outcomes
#axiomsClean pow_scalar_tensor_fp32_host_equivalence
#axiomsClean pow_scalar_tensor_host_failure

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
