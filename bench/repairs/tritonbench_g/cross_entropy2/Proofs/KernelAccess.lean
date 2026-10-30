import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelNative
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelContract
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelAccess

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelAccess
open VeriTile.Triton FP.Structural KernelContract
open CrossEntropy1.TensorModel (Tensor)
open CrossEntropy1.KernelAccess
noncomputable section
set_option maxHeartbeats 4000000

/-- Each actual forward load/store address is backed. The target gather is
required only in its source ownership branch; masked tail lanes never load. -/
theorem forward_backed (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial)
    (launch : Launch p) (pids : LaunchState p launch initial) :
    4*(p.labelsBase+launch.row+1)≤(alloc (Region.cast p.labels)).capacity ∧
    4*(p.lseBase+launch.block*p.rows+launch.row+1)≤(alloc p.lse).capacity ∧
    4*(p.lossBase+launch.block*p.rows+launch.row+1)≤(alloc p.loss).capacity ∧
    4*(p.zLossBase+launch.block*p.rows+launch.row+1)≤(alloc p.zLoss).capacity ∧
    (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
      4*(p.logitsBase+launch.row*p.logitsStride+(launch.block*p.block+lane.val)+1)≤(alloc p.logits).capacity) ∧
    (KernelNative.owns p initial →
      4*(p.logitsBase+launch.row*p.logitsStride+(KernelNative.target p initial-p.classStart).toNat+1)≤(alloc p.logits).capacity) := by
  refine ⟨?_,output_backed _ _ _ _ _ _ _ _ _ views.lseView inputs.lse launch.rowBound launch.blockBound,
    output_backed _ _ _ _ _ _ _ _ _ views.lossView inputs.loss launch.rowBound launch.blockBound,
    output_backed _ _ _ _ _ _ _ _ _ views.zLossView inputs.zLoss launch.rowBound launch.blockBound,?_,?_⟩
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
are backed in the supplied fresh gradient allocation. -/
theorem backward_backed (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : BackwardInputs p views alloc bytes initial)
    (launch : Launch p) :
    4*(p.labelsBase+launch.row+1)≤(alloc (Region.cast p.labels)).capacity ∧
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
    (CrossEntropy1.Native.column p.block launch.block lane.val).toInt=(launch.block*p.block+lane.val : Nat) ∧
    CrossEntropy1.Native.columnMask p.cols p.block launch.block lane.val=decide (launch.block*p.block+lane.val<p.cols) :=
  CrossEntropy1.Native.source_mask p.rows (CrossEntropy1.Math.blocks p.cols p.block) p.block p.cols launch.block lane.val
    valid.geometry launch.blockBound lane.isLt

/-- The source's widened pointer expression names exactly the actual FP32
Tensor element in byte-addressed native memory, with no modular wrap. -/
theorem forward_logit_address (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial)
    (launch : Launch p) (lane : Fin p.block) (active : launch.block*p.block+lane.val<p.cols) :
    (CrossEntropy1.Native.address (alloc p.logits).base 4
      (CrossEntropy1.Native.element p.logitsBase launch.row p.logitsStride p.block launch.block lane.val)).toNat=
      (alloc views.logits.region).base+4*views.logits.index (launch.row*p.cols+(launch.block*p.block+lane.val)) := by
  rw [CrossEntropy1.Native.element_word p.logitsBase launch.row p.logitsStride p.block launch.block lane.val
    p.rows (CrossEntropy1.Math.blocks p.cols p.block) p.cols inputs.parameters.geometry launch.rowBound launch.blockBound lane.isLt]
  rw [matrix_index _ _ _ _ _ _ _ _ views.logitsView launch.rowBound active,views.logitsView.regionEq]
  exact (CrossEntropy1.Native.backed_address (alloc p.logits) _ 4 (by decide)
    (matrix_backed _ _ _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits.toTensorStorage launch.rowBound active)
    (by simpa only [views.logitsView.regionEq] using inputs.logits.addressSpace)).1

theorem forward_output_addresses (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial) (launch : Launch p) :
    (CrossEntropy1.Native.address (alloc p.lse).base 4 (CrossEntropy1.Native.scalarOutput p.lseBase p.rows launch.row launch.block)).toNat=
      (alloc p.lse).base+4*(p.lseBase+launch.block*p.rows+launch.row) ∧
    (CrossEntropy1.Native.address (alloc p.loss).base 4 (CrossEntropy1.Native.scalarOutput p.lossBase p.rows launch.row launch.block)).toNat=
      (alloc p.loss).base+4*(p.lossBase+launch.block*p.rows+launch.row) ∧
    (CrossEntropy1.Native.address (alloc p.zLoss).base 4 (CrossEntropy1.Native.scalarOutput p.zLossBase p.rows launch.row launch.block)).toNat=
      (alloc p.zLoss).base+4*(p.zLossBase+launch.block*p.rows+launch.row) := by
  refine ⟨?_,?_,?_⟩
  · rw [CrossEntropy1.Native.scalar_output_word p.lseBase p.rows launch.row launch.block
      (CrossEntropy1.Math.blocks p.cols p.block) p.block p.cols inputs.parameters.geometry launch.rowBound launch.blockBound]
    exact (CrossEntropy1.Native.backed_address (alloc p.lse) _ 4 (by decide)
      (output_backed _ _ _ _ _ _ _ _ _ views.lseView inputs.lse launch.rowBound launch.blockBound)
      (by simpa only [views.lseView.regionEq] using inputs.lse.addressSpace)).1
  · rw [CrossEntropy1.Native.scalar_output_word p.lossBase p.rows launch.row launch.block
      (CrossEntropy1.Math.blocks p.cols p.block) p.block p.cols inputs.parameters.geometry launch.rowBound launch.blockBound]
    exact (CrossEntropy1.Native.backed_address (alloc p.loss) _ 4 (by decide)
      (output_backed _ _ _ _ _ _ _ _ _ views.lossView inputs.loss launch.rowBound launch.blockBound)
      (by simpa only [views.lossView.regionEq] using inputs.loss.addressSpace)).1
  · rw [CrossEntropy1.Native.scalar_output_word p.zLossBase p.rows launch.row launch.block
      (CrossEntropy1.Math.blocks p.cols p.block) p.block p.cols inputs.parameters.geometry launch.rowBound launch.blockBound]
    exact (CrossEntropy1.Native.backed_address (alloc p.zLoss) _ 4 (by decide)
      (output_backed _ _ _ _ _ _ _ _ _ views.zLossView inputs.zLoss launch.rowBound launch.blockBound)
      (by simpa only [views.zLossView.regionEq] using inputs.zLoss.addressSpace)).1

theorem int32_widen (word : BitVec 32) : (word.signExtend 64).toInt=word.toInt :=
  BitVec.toInt_signExtend_of_le (by decide)

def safeTarget (label start : Int) (cols : Nat) : Int :=
  if start≤label ∧ label<start+(cols : Int) then label else start

def safeShift (label start : Int) (cols : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (safeTarget label start cols)-BitVec.ofInt 64 start

theorem safe_shift_signed (label start : Int) (cols total : Nat)
    (startNonneg : 0≤start) (metadata : start+(cols : Int)≤(total : Int)) (totalBound : total<2^63) :
    (safeShift label start cols).toInt=
      if start≤label ∧ label<start+(cols : Int) then label-start else 0 := by
  by_cases owned : start≤label ∧ label<start+(cols : Int)
  · have word : safeShift label start cols=BitVec.ofInt 64 (label-start) := by
      simp [safeShift,safeTarget,owned,sub_eq_add_neg,BitVec.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg]
    rw [word,if_pos owned]
    apply BitVec.toInt_ofInt_eq_self
    all_goals norm_num at totalBound ⊢
    all_goals omega
  · simp [safeShift,safeTarget,owned]

#axiomsClean int32_widen
#axiomsClean safe_shift_signed

#axiomsClean forward_logit_address
#axiomsClean forward_output_addresses

#axiomsClean forward_backed
#axiomsClean backward_backed
#axiomsClean native_mask
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelAccess
