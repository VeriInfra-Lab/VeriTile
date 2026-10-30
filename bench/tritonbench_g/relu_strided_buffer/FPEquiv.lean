import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeMetadata
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeAccess
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeFP32
import bench.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Constant folding of the unused tile count in both complete branch bodies.
Default floating precision is fp32 (ProfiledRewrite); the pointer element casts,
opaque comparisons, unspecified block padding, masks and grid-stride iterations
are unchanged. Empty numerical domains suffice because only exact Nat arithmetic
changes. Equivalence preserves the entire state, including every memory cell and
execution failure, for arbitrary symbolic strides, dimensions and launch grids. -/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (x y : RegionName) (sx sy n tasks tiles B : Nat)

def oneTile : Program where
  kernel := relu_forward_kernel_rank_1_one_tile_surface x y sx sy n tasks tiles B
  domain := []

def oneTileFolded : Program where
  kernel := Candidate.relu_forward_kernel_rank_1_one_tile_surface x y sx sy n tasks tiles B
  domain := []

def gridStride : Program where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface x y sx sy n tasks tiles B
  domain := []

def gridStrideFolded : Program where
  kernel := Candidate.relu_forward_kernel_rank_1_grid_stride_surface x y sx sy n tasks tiles B
  domain := []

specification one_tile_equiv (_hB : 0 < B) :
    oneTile x y sx sy n tasks tiles B ≡[assumptions]
      oneTileFolded x y sx sy n tasks tiles B := by
  equiv_decompose
  all_goals fp_prove

specification grid_stride_equiv (_hB : 0 < B) :
    gridStride x y sx sy n tasks tiles B ≡[assumptions]
      gridStrideFolded x y sx sy n tasks tiles B := by
  equiv_decompose
  all_goals fp_prove
end

#print_fp_assumptions one_tile_equiv
#print_fp_assumptions grid_stride_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Equivalence

namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
open VeriTile.Triton FP.Structural HostPlan SignedView SignedGrid
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Complete original FP32 host result, finite source access and entry-relative
cell/byte frame. Source precision is FP32; all 2^32 input words are allowed.
The exact result is ordered `x > 0 ? x : +0`: both signed zeros and every NaN
select +0, positive subnormals/infinity retain their input bits. The independent
Real entry uses max(0,x) and makes no equality claim for non-real FP values.

Domain: nonempty rank one, n<2^31, both signed views valid, nonzero output
stride, disjoint input/output footprints, same CUDA device and successful guard.
NativeBuffers supplies normal allocation provenance, alignment and capacity;
all launched indices/masks and all four active bytes are then proved exact.
There are no numerical assumptions or result-producing runtime callbacks. -/
specification relu_strided_buffer_fp32 (rest : Algebra (FP.Value .fp32))
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
    (s : State (FP.Value .fp32)) (xs : Nat → FP.Value .fp32)
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    ∃ final, runHost (fp32Backend rest) input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (fp32Result xs i)) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) ∧
      NativeAccess input output n sx sy buffers ∧
      SourceAddressLowering input output n sx sy buffers ∧
      (∀ i, i < n → ∀ byte, byte < 4 →
        memoryByte final output.region ((output.view sy).address i) byte =
          wordByte (fp32Result xs i) byte) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        ∀ byte, byte < 4 → memoryByte final r a byte = memoryByte s r a byte) ∧
      (∀ i j u v, i < n → j < n → u < 4 → v < 4 →
        buffers.inputBase + 4 * (input.view sx).address i + u ≠
          buffers.outputBase + 4 * (output.view sy).address j + v) := by
  obtain ⟨final, run, outputValues, inputValues, frame, bounds⟩ :=
    fp32_complete rest input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
      hin hout hstride hsep s xs values
  refine ⟨final, run, outputValues, inputValues, frame, bounds,
    native_access input output positive bounded sx sy hin hout buffers,
    source_address_lowering input output positive bounded sx sy hin hout buffers, ?_, ?_, ?_⟩
  · intro i hi byte _
    exact output_bytes final output.region ((output.view sy).address i)
      (fp32Result xs i) (outputValues i hi) byte
  · intro r a outside byte _
    exact frame_bytes s final r a (frame r a outside) byte
  · intro i j u v hi hj hu hv
    exact input_output_bytes_disjoint input output sx sy hin hout hsep buffers hi hj hu hv

/-- Empty input retains the original host failure, even with an unsuccessful
CUDA guard: the tile heuristic runs first and no memory is modified. -/
theorem relu_strided_buffer_fp32_empty (rest : Algebra (FP.Value .fp32))
    (input output : Buffer) (guardSucceeds : Bool) (s : State (FP.Value .fp32))
    (hin : input.shape = [0]) (hout : output.shape = [0]) :
    runHost (fp32Backend rest) input output guardSucceeds s = .hostFailure .tileDivision s :=
  StructuralHost.empty_failure _ input output guardSucceeds s hin hout

/-- All pre-launch source failures preserve the complete entry state. This
includes shape assertion, guard failure and malformed metadata access. -/
theorem relu_strided_buffer_host_failure (rest : Algebra (FP.Value .fp32))
    (input output : Buffer) (guardSucceeds : Bool) (s : State (FP.Value .fp32))
    (failure : Failure) (failed : prepare input output guardSucceeds = .error failure) :
    runHost (fp32Backend rest) input output guardSucceeds s = .hostFailure failure s :=
  StructuralHost.failure_preserves_memory _ input output guardSucceeds s failure failed

#axiomsClean relu_strided_buffer_fp32
#axiomsClean relu_strided_buffer_fp32_empty
#axiomsClean relu_strided_buffer_host_failure
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
