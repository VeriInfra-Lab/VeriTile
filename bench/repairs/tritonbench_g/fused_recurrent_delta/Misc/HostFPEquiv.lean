import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.FPEquivPublic
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
noncomputable section
open Admission Host

/-- Exact source FP bytes and actual output/gradient objects. Rejection leaves
the corresponding source memory unchanged. Only four fixed-arity numerical
instruction correspondences are conditional; structural admission is checked. -/
theorem checked_fp_returned (c : ABI.Call) (device : Nat) (inputs : List Region)
    (outputIsBool scaleAccepted : Bool) (objects : Objects)
    (ok : check c device (observations inputs objects) outputIsBool scaleAccepted=Bool.true)
    (fa : allocationCheck c objects inputs (forwardAllocations c)=Bool.true)
    (ba : allocationCheck c objects inputs (backwardAllocations c)=Bool.true)
    (cotangents : cotangentCheck c inputs=Bool.true)
    (gf gb : Trees) (actual : Native.Instructions) (ports : Native.Obligations actual)
    (pf pb : Native.Policy) (initial : PhysicalLayout.Bytes) :
    let fw := forwardFP c device inputs outputIsBool scaleAccepted objects gf actual pf initial
    let bw := backwardFP c device inputs outputIsBool scaleAccepted Bool.false (saved c objects) objects gb actual pb fw.memory
    fw.value=.ok (outputTuple c objects) ∧ bw.value=.ok (gradientTuple c objects) ∧
    (gradientTuple c objects).length=7 ∧
    fw.memory=(runFP c gf gb Native.software pf pb initial).forwardMemory ∧
    bw.memory=(runFP c gf gb Native.software pf pb initial).backwardMemory := by
  simp [forwardFP,backwardFP,ok,fa,ba,cotangents,saved,backwardCheck,gradientTuple,runFP,
    PhysicalLayout.forward_transport c.shape (ABI.flags c) gf c.scale.launch actual ports pf,
    PhysicalLayout.backward_transport c.shape (ABI.flags c) gb c.scale.launch actual ports pb]

abbrev checked_global_byte_frame := @Host.global_frame
abbrev checked_fp_forward_frame := @Host.forward_frame
abbrev checked_fp_backward_frame := @Host.backward_frame
abbrev checked_fp_rejected := @Host.forward_rejected
abbrev checked_fp_backward_rejected := @Host.backward_rejected
abbrev checked_source_indices := @Admission.checked_index
abbrev checked_source_addresses := @Admission.checked_active_bytes

#specNonCircular Host.forwardFP avoiding [WholeSource.forwardSoftwareReference]
#specNonCircular Host.backwardFP avoiding [WholeSource.backwardSoftwareReference]
#axiomsClean checked_fp_returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta
