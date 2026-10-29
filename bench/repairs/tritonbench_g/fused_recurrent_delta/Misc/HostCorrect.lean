import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.CorrectPublic
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
noncomputable section
open Admission Host

/-- Independent Real correctness and complete VJP of a completed guarded call.
The saved snapshot is the actual forward-mutated memory used by backward. -/
theorem checked_returned (c : ABI.Call) (device : Nat) (inputs : List Region)
    (outputIsBool scaleAccepted : Bool) (objects : Objects)
    (ok : check c device (observations inputs objects) outputIsBool scaleAccepted=Bool.true)
    (fa : allocationCheck c objects inputs (forwardAllocations c)=Bool.true)
    (ba : allocationCheck c objects inputs (backwardAllocations c)=Bool.true)
    (cotangents : cotangentCheck c inputs=Bool.true)
    (gf gb : Trees) (trees : RealBackward.Legal c.shape gf gb) (initial : Memory ℝ) :
    let fw := forwardReal c device inputs outputIsBool scaleAccepted objects gf initial
    let bw := backwardReal c device inputs outputIsBool scaleAccepted Bool.false (saved c objects) objects gb fw.memory
    fw.value=.ok (outputTuple c objects) ∧ bw.value=.ok (gradientTuple c objects) ∧
    ForwardClaim c.shape (ABI.flags c) (ABI.realScale c) (prepareBeta c.shape (ABI.flags c) Real.ops initial) fw.memory ∧
    BackwardClaim c.shape (ABI.flags c) (ABI.realScale c) (prepareBeta c.shape (ABI.flags c) Real.ops initial) bw.memory ∧
    MemoryProof.Preserves (MemoryProof.ForwardCallWrites c.shape (ABI.flags c)) initial fw.memory ∧
    MemoryProof.Preserves (MemoryProof.BackwardWrites c.shape (ABI.flags c)) fw.memory bw.memory := by
  have facts := correct c.shape (ABI.flags c) gf gb trees (ABI.realScale c) initial
  simpa only [forwardReal,backwardReal,ok,fa,ba,cotangents,saved,backwardCheck,
    Bool.not_false,Bool.true_and,↓reduceIte,true_and] using facts

#axiomsClean checked_returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta
