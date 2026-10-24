import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Contract
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeAccess

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open HostPlan SignedView SignedGrid
noncomputable section

/-- Ordinary storage allocation/provenance and source metadata interpretation.
Alignment, shape, bounds, dtype, aliasing and stride restrictions are checked. -/
structure AllocationAPI (x y : ViewMetadata) : Prop where
  inputAllocation : SignedByteAddress.AllocationDomain x.allocationBase x.capacity
  outputAllocation : SignedByteAddress.AllocationDomain y.allocationBase y.capacity
  sameAllocation : x.buffer.region = y.buffer.region → x.allocationBase = y.allocationBase
  separateAllocations : x.buffer.region ≠ y.buffer.region →
    x.allocationBase+4*x.capacity ≤ y.allocationBase ∨ y.allocationBase+4*y.capacity ≤ x.allocationBase
  inputDevice : x.buffer.deviceIndex = some x.device.index
  outputDevice : y.buffer.deviceIndex = some y.device.index

def nativeBuffers (x y : ViewMetadata) (h : InputContract x y) (api : AllocationAPI x y) :
    Whole.NativeBuffers x.buffer y.buffer x.capacity y.capacity where
  inputBase := x.allocationBase
  outputBase := y.allocationBase
  inputAlignment := h.input.alignment
  outputAlignment := h.output.alignment
  inputAllocation := api.inputAllocation
  outputAllocation := api.outputAllocation
  sameAllocation := api.sameAllocation
  separateAllocations := api.separateAllocations

theorem ViewContract.valid (x : ViewMetadata) (h : ViewContract x) :
    (x.buffer.view x.stride).Valid x.n x.capacity :=
  (endpoint_bounds_iff x h.extent.1).mp h.bounds

theorem input_output_footprints (x y : ViewMetadata) (h : InputContract x y) (api : AllocationAPI x y) :
    FootprintsDisjoint (x.buffer.view x.stride) (y.buffer.view y.stride) x.n := by
  intro i hi j hj
  by_cases regions : x.buffer.region ≠ y.buffer.region
  · exact Or.inl regions
  · right
    intro collision
    have base := api.sameAllocation (not_ne_iff.mp regions)
    have inputValid := ViewContract.valid x h.input
    have outputValid := ViewContract.valid y h.output
    have hj' : j < y.n := h.shape ▸ hj
    have signed := congrArg (fun a : Nat => (a : Int)) collision
    dsimp only at signed
    rw [SignedView.address_cast _ (inputValid i hi).1,
      SignedView.address_cast _ (outputValid j hj').1] at signed
    simp only [Buffer.view,SignedView.View.signedAddress] at signed
    apply h.disjoint i hi j hj
    unfold ViewMetadata.start
    rw [base]
    omega

/-- The Python guard checks and overlap test use exactly these observed cells. -/
theorem pointer_cell_start (x : ViewMetadata) (h : ViewContract x) :
    x.pointer / 4 = x.start ∧ (x.pointer-(x.allocationBase : Int))/4 = x.buffer.base := by
  rw [h.pointer]
  unfold ViewMetadata.start
  constructor <;> omega

theorem actual_pointer (x : ViewMetadata) (h : ViewContract x)
    (allocation : SignedByteAddress.AllocationDomain x.allocationBase x.capacity) :
    (SignedByteAddress.dataPtr64 x.allocationBase (x.buffer.view x.stride)).toNat = x.pointer.toNat := by
  have valid := ViewContract.valid x h
  rw [SignedByteAddress.data_ptr_exact x.allocationBase _ valid allocation h.extent.1,h.pointer]
  have nonneg := (valid 0 h.extent.1).1
  simp only [Buffer.view,SignedView.View.signedAddress,Nat.cast_zero,zero_mul,add_zero] at nonneg
  simp [Buffer.view,Int.toNat_add,Int.toNat_mul,nonneg]


#axiomsClean input_output_footprints
#axiomsClean actual_pointer
#axiomsClean pointer_cell_start
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
