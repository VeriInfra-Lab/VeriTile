import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelProfile
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelLaunch

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelBoundary
open VeriTile.Triton FP.Structural KernelSource KernelContract KernelContract.Storage
noncomputable section
set_option maxHeartbeats 4000000

/-- The exact argument pointers used by a forward program name the actual
Tensor elements and signed native source offsets. Chunking changes only the
LSE output index and source column range. -/
def ForwardAddresses (p : Parameters) (views : ForwardViews p chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words) (launch : Launch p chunks) : Prop :=
  (KernelAccess.pointer64 8 inputs.labelsPointer launch.row).toNat=
    (alloc (Region.cast p.labels)).base+8*views.labels.index launch.row ∧
  (KernelAccess.pointer64 4 inputs.lossPointer launch.row).toNat=
    (alloc p.loss).base+4*views.loss.index launch.row ∧
  (KernelAccess.pointer64 4 inputs.lsePointer (launch.row*chunks+launch.block)).toNat=
    (alloc p.lse).base+4*views.lse.index (launch.row*chunks+launch.block) ∧
  (∀ col, col<p.vocab →
    (KernelAccess.pointer64 4 inputs.logitsPointer (launch.row*p.logitsStride+col)).toNat=
      (alloc p.logits).base+4*views.logits.index (launch.row*p.vocab+col) ∧
    (BitVec.ofNat 64 (launch.row*p.logitsStride+col)).toInt=(launch.row*p.logitsStride+col : Nat))

 theorem forward_addresses (p : Parameters) (views : ForwardViews p chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words) (launch : Launch p chunks) :
    ForwardAddresses p views alloc bytes words inputs launch := by
  have hi : launch.row*chunks+launch.block<p.rows*chunks := by
    have bound := Nat.mul_le_mul_right chunks (Nat.succ_le_of_lt launch.rowBound)
    have hb := launch.blockBound
    simp only [Nat.succ_mul] at bound
    omega
  refine ⟨KernelAccess.flat_pointer _ _ _ _ _ _ _ views.labelsView inputs.labels.toTensorStorage (by omega) launch.rowBound,
    KernelAccess.flat_pointer _ _ _ _ _ _ _ views.lossView inputs.loss (by omega) launch.rowBound,
    KernelAccess.flat_pointer _ _ _ _ _ _ _ views.lseView inputs.lse (by omega) hi,?_⟩
  intro col hc
  exact KernelAccess.matrix_pointer _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits.toTensorStorage launch.rowBound hc

/-- The all-word compiler source entry really contains the word assembled
from the actual Tensor's four little-endian allocation bytes. -/
 theorem forward_logit_leaf (p : Parameters) (views : ForwardViews p chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words) (launch : Launch p chunks)
    (col : Nat) (hc : col<p.vocab) :
    ((KernelProfile.entry p words).mem p.logits (launch.row*p.logitsStride+col)).read .real=
      .leaf p.logits (launch.row*p.logitsStride+col)
        (wordAt (bytes p.logits) (4*views.logits.index (launch.row*p.vocab+col))) := by
  rw [KernelProfile.entry_word p words p.logits _ inputs.labelSeparation.1]
  rw [KernelAccess.logits_words _ _ _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits launch.rowBound hc]

 theorem forward_label_bytes (p : Parameters) (views : ForwardViews p chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words) (launch : Launch p chunks)
    (pids : LaunchState launch words) :
    label p (KernelProfile.entry p words)=intAt (bytes (Region.cast p.labels)) (8*views.labels.index launch.row) := by
  rw [KernelProfile.entry_label,label,pids.1]
  exact KernelAccess.labels_words _ _ _ _ _ _ _ _ views.labelsView inputs.labels launch.rowBound

 def BackwardAddresses (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words) (launch : Launch p blocks) : Prop :=
  (KernelAccess.pointer64 8 inputs.labelsPointer launch.row).toNat=
    (alloc (Region.cast p.labels)).base+8*views.labels.index launch.row ∧
  (KernelAccess.pointer64 4 inputs.lsePointer launch.row).toNat=
    (alloc p.lse).base+4*views.lse.index launch.row ∧
  (KernelAccess.pointer64 4 inputs.upstreamPointer (launch.row*p.upstreamStride)).toNat=
    (alloc p.upstream).base+4*views.upstream.index launch.row ∧
  (∀ col, col<p.vocab →
    (KernelAccess.pointer64 4 inputs.logitsPointer (launch.row*p.logitsStride+col)).toNat=
      (alloc p.logits).base+4*views.logits.index (launch.row*p.vocab+col) ∧
    (BitVec.ofNat 64 (launch.row*p.logitsStride+col)).toInt=(launch.row*p.logitsStride+col : Nat))

 theorem backward_addresses (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words) (launch : Launch p blocks) :
    BackwardAddresses p views alloc bytes words inputs launch := by
  refine ⟨KernelAccess.flat_pointer _ _ _ _ _ _ _ views.labelsView inputs.labels.toTensorStorage (by omega) launch.rowBound,
    KernelAccess.flat_pointer _ _ _ _ _ _ _ views.lseView inputs.lse.toTensorStorage (by omega) launch.rowBound,?_,?_⟩
  · have hi : launch.row<views.upstream.numel := by
      simpa only [VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Tensor.numel,
        views.upstreamView.shape,List.prod_cons,List.prod_nil,Nat.mul_one] using launch.rowBound
    simpa only [views.upstreamView.regionEq] using KernelAccess.argument_pointer alloc views.upstream 4
      inputs.upstreamPointer launch.row (launch.row*p.upstreamStride) inputs.upstream.toTensorStorage
      (by omega) hi (vector_index _ _ _ _ _ views.upstreamView launch.rowBound)
  · intro col hc
    exact KernelAccess.matrix_pointer _ _ _ _ _ _ _ _ _ views.logitsView inputs.logits.toTensorStorage launch.rowBound hc

#axiomsClean forward_addresses
#axiomsClean backward_addresses
#axiomsClean forward_logit_leaf
#axiomsClean forward_label_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelBoundary
