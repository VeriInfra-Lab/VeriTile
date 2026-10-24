import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Contract

/-! A single physical FP32-cell memory for the admitted signed views. Storage
objects can overlap or have different bases, including DLPack aliases. Only
actual active byte spans must be separated, exactly as the Python guard checks.
The existing structural memory and signed-grid proof are reused unchanged. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.Physical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
open HostPlan SignedView SignedGrid
noncomputable section

/-- One native address space, indexed by absolute aligned FP32 cells. -/
def region : RegionName := "memory"
def cellBase (v : ViewMetadata) : Int := ((v.storageBase : Int) + v.delta) / 4
def capacity : Nat := 2^62
def buffer (v : ViewMetadata) : Buffer :=
  ⟨region,cellBase v,v.shape,v.strides,some v.device⟩
@[simp] theorem buffer_region (v : ViewMetadata) : (buffer v).region = region := rfl
def address (v : ViewMetadata) (i : Nat) : Nat := ((buffer v).view (stride v)).address i

theorem base_bytes (v : ViewMetadata) (h : ViewContract v) :
    4 * cellBase v = (v.storageBase : Int) + v.delta := by
  have storageAligned : (v.storageBase : Int) % 4 = 0 := by exact_mod_cast h.aligned.1
  have viewAligned := h.aligned.2
  unfold cellBase
  omega

theorem data_pointer (v : ViewMetadata) (h : ViewContract v) :
    SignedByteAddress.dataPtr64 0 ((buffer v).view (stride v)) =
      BitVec.ofInt 64 ((v.storageBase : Int) + v.delta) := by
  simpa [SignedByteAddress.dataPtr64,buffer,Buffer.view,BitVec.ofInt_mul] using
    congrArg (BitVec.ofInt 64) (base_bytes v h)

theorem view_valid (v : ViewMetadata) (h : ViewContract v) :
    ((buffer v).view (stride v)).Valid (count v) capacity := by
  intro i hi
  obtain ⟨lo,up⟩ := coordinate_bounds v i hi
  obtain ⟨lowerBound,upperBound⟩ := h.bounds (by omega)
  have allocation : (v.storageBase : Int) + v.storageBytes ≤ 2^64 := by
    exact_mod_cast h.allocation
  have exactBase := base_bytes v h
  change 0 ≤ cellBase v + (i : Int) * stride v ∧
    cellBase v + (i : Int) * stride v < (capacity : Int)
  norm_num [capacity]
  constructor <;> nlinarith

/-- Alignment and the complete signed sum precede division/toNat. The proof
cell is exactly the checked view pointer plus its signed byte stride. -/
theorem cell_bytes (v : ViewMetadata) (h : ViewContract v) (i : Nat) (hi : i < count v) :
    4 * (address v i : Int) = (v.storageBase : Int) + v.delta + 4 * (i : Int) * stride v := by
  unfold address
  rw [address_cast _ ((view_valid v h) i hi).1]
  change 4 * (cellBase v + (i : Int) * stride v) = _
  have exactBase := base_bytes v h
  nlinarith

/-- Actual backing containment, separately from the uint64 coordinate range. -/
def InsideBacking (v : ViewMetadata) (i : Nat) : Prop :=
    (v.storageBase : Int) ≤ 4 * (address v i : Int) ∧
      4 * (address v i : Int) + 4 ≤ (v.storageBase : Int) + v.storageBytes

/-- Each physical access stays inside its own observed backing span, without
requiring the two backing storage objects to be equal or disjoint. -/
theorem backing_bounds (v : ViewMetadata) (h : ViewContract v) (i : Nat) (hi : i < count v) :
    InsideBacking v i := by
  unfold InsideBacking
  rw [cell_bytes v h i hi]
  obtain ⟨lo,up⟩ := coordinate_bounds v i hi
  obtain ⟨lowerBound,upperBound⟩ := h.bounds (by omega)
  constructor <;> linarith

theorem footprint_separation (f : Bool) (x y : ViewMetadata) (h : InputContract f x y) :
    FootprintsDisjoint ((buffer x).view (stride x)) ((buffer y).view (stride y)) (count x) := by
  intro i hi j hj
  right
  intro collision
  have hj' : j < count y := by simpa [←h.shape] using hj
  have xb := cell_bytes x h.input i hi
  have yb := cell_bytes y h.output j hj'
  change address x i = address y j at collision
  rw [collision] at xb
  obtain ⟨xl,xu⟩ := coordinate_bounds x i hi
  obtain ⟨yl,yu⟩ := coordinate_bounds y j hj'
  have spans := h.separation
  simp only [beginByte,endByte] at spans
  rcases spans with xy | yx <;> nlinarith

/-- The zero-origin uint64 address space is a coordinate system, not a new
allocation premise. Actual backing containment is `backing_bounds` above. -/
def native_buffers (x y : ViewMetadata) :
    NativeBuffers (buffer x) (buffer y) capacity capacity where
  inputBase := 0
  outputBase := 0
  inputAlignment := by decide
  outputAlignment := by decide
  inputAllocation := by norm_num [SignedByteAddress.AllocationDomain,
    VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.ByteDomain,capacity]
  outputAllocation := by norm_num [SignedByteAddress.AllocationDomain,
    VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.ByteDomain,capacity]
  sameAllocation := fun _ => rfl
  separateAllocations := by intro impossible; exact False.elim (impossible rfl)

#axiomsClean base_bytes
#axiomsClean data_pointer
#axiomsClean view_valid
#axiomsClean cell_bytes
#axiomsClean backing_bounds
#axiomsClean footprint_separation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.Physical
