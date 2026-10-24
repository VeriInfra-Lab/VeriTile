import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.CheckedReal
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.WholeUnsplit
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.WholeSplit
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Aggregation

namespace VeriTile.Bench.Repairs.CrossEntropyOps.HostCorrect

/-!
Independent Real target for the complete repaired Python wrapper. For global
C>0, u[j]=scale*x[j] and L=log(sum exp u), loss is
L-(1-s)*u[target]-s*sum(u)/C+z*L*L; logging loss is z*L*L; gradient is
(upstream*scale)*((1+2*z*L)*exp(u[j]-L)-indicator*(1-s)-s/C).
Ignored rows select exact zero. Logging loss is nondifferentiable.

The declared application domain is FP32 logits/upstream, int64 labels, matching
logical shapes/devices, valid nonignored labels, s in [0,1], z>=0, arbitrary Real
scale (including zero/negative), and equal-width distributed ranks with matched
group/scalar/label metadata. Empty row batches are admitted. Some Real lemmas
prove a stronger scalar range. Ordinary runtime errors, explicit zero-class /
overlapping-inplace errors and saved-version errors remain API outcomes.

The public whole-call results below consume actual successful forward and
version-checked backward calls. They derive returned tensors, fresh allocation
separation, saved labels/global LSE, shape and inplace return identity from
ordinary tensor-library contracts. Neither a correct saved LSE nor a correct
peer result is a premise. Native byte results are in the independent FP entry.
-/
export Spec (loss zLoss gradient)
export Host (unsplit_forward_backward split_forward_backward
  backward_entry_modified backward_entry_success logging_gradient_unused backward_tuple_length)
export RealGrid (unsplit_outputs unsplit_roundtrip_gradient)
export Aggregation (actual_combined_lse actual_combined_loss)
export Grid (forward_complete_frame backward_complete_frame)

#axiomsClean unsplit_forward_backward
#axiomsClean split_forward_backward
#axiomsClean unsplit_outputs
#axiomsClean unsplit_roundtrip_gradient
#axiomsClean actual_combined_lse
#axiomsClean actual_combined_loss
#axiomsClean forward_complete_frame
#axiomsClean backward_complete_frame
#axiomsClean backward_entry_modified
#axiomsClean backward_entry_success
#specNonCircular Spec.loss avoiding [Host.forward,Host.backward,Source.forward,Source.backward]
#specNonCircular Spec.zLoss avoiding [Host.forward,Host.backward,Source.forward,Source.backward]
#specNonCircular Spec.gradient avoiding [Host.forward,Host.backward,Source.forward,Source.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CrossEntropyOps.HostCorrect
