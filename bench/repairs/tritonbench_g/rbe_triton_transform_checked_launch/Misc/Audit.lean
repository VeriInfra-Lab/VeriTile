import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Misc.TensorBridge
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Correct
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.FPEquiv
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelAudit

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Audit
open VeriTile Triton Torch
open VeriTile.Bench.Repairs
noncomputable section

def metadata (b m k : Nat) : TensorMetadata :=
  ⟨.floating .fp32,.strided,⟨"cuda",0⟩,[b,m,k],Bool.true,Bool.false,BitVec.ofNat 64 4096⟩

theorem original_counterexample_rejected :
    RbeTritonTransform.Contract.checkInputs (metadata 1 131071 1) ⟨Bool.true,0⟩=.ok () ∧
    Contract.checkInputs (metadata 1 131071 1) ⟨Bool.true,0⟩=.error .grid := by decide

theorem grid_boundary :
    Contract.checkInputs (metadata 1 131070 1) ⟨Bool.true,0⟩=.ok () ∧
    Contract.checkInputs (metadata 1 131071 1) ⟨Bool.true,0⟩=.error .grid := by decide

theorem batch_boundary :
    Contract.checkInputs (metadata (2^31-1) 1 1) ⟨Bool.true,0⟩=.ok () ∧
    Contract.checkInputs (metadata (2^31) 1 1) ⟨Bool.true,0⟩=.error .batch := by decide

theorem empty_large_batch_unchanged :
    Contract.checkInputs (metadata (2^31) 0 1) ⟨Bool.true,2^63-1⟩=.ok () ∧
    Contract.checkInputs (metadata (2^31) 131071 0) ⟨Bool.true,2^63-1⟩=.ok () := by decide

#guard_msgs (drop info) in
#specNonCircular Input avoiding [RbeTritonTransform.Selected.source,
  RbeTritonTransform.Runtime.run,FP.Structural.exec]
#axiomsClean Contract.checkInputs_iff
#axiomsClean Contract.empty_domain_unchanged
#axiomsClean Input.supported
#axiomsClean TensorBridge.kernel_correct
#axiomsClean TensorBridge.kernel_native_equivalence
#axiomsClean Correct.kernel_correct
#axiomsClean FPEquiv.kernel_native_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Audit
