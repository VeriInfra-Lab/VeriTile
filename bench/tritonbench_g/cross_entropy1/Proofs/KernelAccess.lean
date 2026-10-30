import bench.tritonbench_g.cross_entropy1.Proofs.KernelNative
import bench.tritonbench_g.cross_entropy1.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelAccess
open VeriTile.Triton FP.Structural KernelContract
open TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

theorem matrix_backed (alloc : RegionName → Allocation) (t : Tensor)
    (r : RegionName) (base rows cols stride pointer bytes row col : Nat)
    (view : MatrixView t r base rows cols stride) (storage : TensorStorage alloc t bytes pointer)
    (hr : row<rows) (hc : col<cols) :
    bytes*(base+row*stride+col+1)≤(alloc r).capacity := by
  have bound := storage.backed (row*cols+col) (matrix_numel t r base rows cols stride row col view hr hc)
  simpa only [matrix_index _ _ _ _ _ _ _ _ view hr hc,view.regionEq] using bound

theorem vector_backed (alloc : RegionName → Allocation) (t : Tensor)
    (r : RegionName) (base count stride pointer bytes i : Nat)
    (view : VectorView t r base count stride) (storage : TensorStorage alloc t bytes pointer)
    (hi : i<count) : bytes*(base+i*stride+1)≤(alloc r).capacity := by
  have bound := storage.backed i (by simpa [Tensor.numel,view.shape] using hi)
  simpa only [vector_index _ _ _ _ _ _ view hi,view.regionEq] using bound

theorem output_backed (alloc : RegionName → Allocation) (t : Tensor)
    (r : RegionName) (base rows blocks pointer row block : Nat)
    (view : ScalarOutputView t r base rows blocks) (storage : TensorStorage alloc t 4 pointer)
    (hr : row<rows) (hb : block<blocks) :
    4*(base+block*rows+row+1)≤(alloc r).capacity := by
  have hi : block*rows+row<t.numel := by
    by_cases many : 1<blocks
    · have bound := Nat.mul_le_mul_right rows (Nat.succ_le_of_lt hb)
      simp only [Nat.succ_mul] at bound
      simp only [Tensor.numel,view.shape,if_pos many,List.prod_cons,List.prod_nil,Nat.mul_one]
      omega
    · have zero : block=0 := by omega
      simpa [Tensor.numel,view.shape,many,zero] using hr
  have bound := storage.backed (block*rows+row) hi
  simpa only [output_index _ _ _ _ _ _ _ view hr hb,view.regionEq] using bound

/-- Each actual forward load/store address is backed. The target gather is
required only in its source ownership branch; masked tail lanes never load. -/
theorem forward_backed (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial)
    (launch : Launch p) (pids : LaunchState p launch initial) :
    8*(p.labelsBase+launch.row+1)≤(alloc (Region.cast p.labels)).capacity ∧
    4*(p.lseBase+launch.block*p.rows+launch.row+1)≤(alloc p.lse).capacity ∧
    4*(p.lossBase+launch.block*p.rows+launch.row+1)≤(alloc p.loss).capacity ∧
    (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
      4*(p.logitsBase+launch.row*p.logitsStride+(launch.block*p.block+lane.val)+1)≤(alloc p.logits).capacity) ∧
    (KernelNative.owns p initial →
      4*(p.logitsBase+launch.row*p.logitsStride+(KernelNative.target p initial-p.classStart).toNat+1)≤(alloc p.logits).capacity) := by
  refine ⟨?_,output_backed _ _ _ _ _ _ _ _ _ views.lseView inputs.lse launch.rowBound launch.blockBound,
    output_backed _ _ _ _ _ _ _ _ _ views.lossView inputs.loss launch.rowBound launch.blockBound,?_,?_⟩
  · simpa only [Nat.mul_one] using vector_backed _ _ _ _ _ _ _ _ _ views.labelsView inputs.labels.toTensorStorage launch.rowBound
  · intro lane active
    exact matrix_backed _ _ _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits.toTensorStorage launch.rowBound active
  · intro hit
    have active : (KernelNative.target p initial-p.classStart).toNat<p.cols := by
      unfold KernelNative.owns at hit
      have nonneg : (0 : Int)≤KernelNative.target p initial-p.classStart := le_trans (by positivity) hit.1
      have upper := lt_of_lt_of_le hit.2 (min_le_left _ _)
      omega
    exact matrix_backed _ _ _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits.toTensorStorage launch.rowBound active

/-- Backward's initial scalar reads and all active logit/gradient addresses
are backed even when the gradient shares the logit storage. -/
theorem backward_backed (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : BackwardInputs p views alloc bytes initial)
    (launch : Launch p) :
    8*(p.labelsBase+launch.row+1)≤(alloc (Region.cast p.labels)).capacity ∧
    4*(p.lseBase+launch.row+1)≤(alloc p.lse).capacity ∧
    4*(p.upstreamBase+launch.row*p.upstreamStride+1)≤(alloc p.upstream).capacity ∧
    (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
      4*(p.logitsBase+launch.row*p.logitsStride+(launch.block*p.block+lane.val)+1)≤(alloc p.logits).capacity ∧
      4*(p.gradientBase+launch.row*p.gradientStride+(launch.block*p.block+lane.val)+1)≤(alloc p.gradient).capacity) := by
  refine ⟨?_,?_,?_,?_⟩
  · simpa only [Nat.mul_one] using vector_backed _ _ _ _ _ _ _ _ _ views.labelsView inputs.labels.toTensorStorage launch.rowBound
  · simpa only [Nat.mul_one] using vector_backed _ _ _ _ _ _ _ _ _ views.lseView inputs.lse.toTensorStorage launch.rowBound
  · exact vector_backed _ _ _ _ _ _ _ _ _ views.upstreamView inputs.upstream.toTensorStorage launch.rowBound
  · intro lane active
    exact ⟨matrix_backed _ _ _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits.toTensorStorage launch.rowBound active,
      matrix_backed _ _ _ _ _ _ _ _ _ _ _ views.gradientView inputs.gradient launch.rowBound active⟩

/-- Exact widened int64 columns and masks for every launched lane. -/
theorem native_mask (p : Parameters) (valid : ParametersValid p) (launch : Launch p) (lane : Fin p.block) :
    (Native.column p.block launch.block lane.val).toInt=(launch.block*p.block+lane.val : Nat) ∧
    Native.columnMask p.cols p.block launch.block lane.val=decide (launch.block*p.block+lane.val<p.cols) :=
  Native.source_mask p.rows (Math.blocks p.cols p.block) p.block p.cols launch.block lane.val
    valid.geometry launch.blockBound lane.isLt

/-- The source's widened pointer expression names exactly the actual FP32
Tensor element in byte-addressed native memory, with no modular wrap. -/
theorem forward_logit_address (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial)
    (launch : Launch p) (lane : Fin p.block) (active : launch.block*p.block+lane.val<p.cols) :
    (Native.address (alloc p.logits).base 4
      (Native.element p.logitsBase launch.row p.logitsStride p.block launch.block lane.val)).toNat=
      (alloc views.logits.region).base+4*views.logits.index (launch.row*p.cols+(launch.block*p.block+lane.val)) := by
  rw [Native.element_word p.logitsBase launch.row p.logitsStride p.block launch.block lane.val
    p.rows (Math.blocks p.cols p.block) p.cols inputs.parameters.geometry launch.rowBound launch.blockBound lane.isLt]
  rw [matrix_index _ _ _ _ _ _ _ _ views.logitsView launch.rowBound active,views.logitsView.regionEq]
  exact (Native.backed_address (alloc p.logits) _ 4 (by decide)
    (matrix_backed _ _ _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits.toTensorStorage launch.rowBound active)
    (by simpa only [views.logitsView.regionEq] using inputs.logits.addressSpace)).1

theorem forward_output_addresses (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial) (launch : Launch p) :
    (Native.address (alloc p.lse).base 4 (Native.scalarOutput p.lseBase p.rows launch.row launch.block)).toNat=
      (alloc p.lse).base+4*(p.lseBase+launch.block*p.rows+launch.row) ∧
    (Native.address (alloc p.loss).base 4 (Native.scalarOutput p.lossBase p.rows launch.row launch.block)).toNat=
      (alloc p.loss).base+4*(p.lossBase+launch.block*p.rows+launch.row) := by
  constructor
  · rw [Native.scalar_output_word p.lseBase p.rows launch.row launch.block
      (Math.blocks p.cols p.block) p.block p.cols inputs.parameters.geometry launch.rowBound launch.blockBound]
    exact (Native.backed_address (alloc p.lse) _ 4 (by decide)
      (output_backed _ _ _ _ _ _ _ _ _ views.lseView inputs.lse launch.rowBound launch.blockBound)
      (by simpa only [views.lseView.regionEq] using inputs.lse.addressSpace)).1
  · rw [Native.scalar_output_word p.lossBase p.rows launch.row launch.block
      (Math.blocks p.cols p.block) p.block p.cols inputs.parameters.geometry launch.rowBound launch.blockBound]
    exact (Native.backed_address (alloc p.loss) _ 4 (by decide)
      (output_backed _ _ _ _ _ _ _ _ _ views.lossView inputs.loss launch.rowBound launch.blockBound)
      (by simpa only [views.lossView.regionEq] using inputs.loss.addressSpace)).1

#axiomsClean forward_logit_address
#axiomsClean forward_output_addresses

#axiomsClean forward_backed
#axiomsClean backward_backed
#axiomsClean native_mask
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelAccess
