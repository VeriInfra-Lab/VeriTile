import bench.repairs.tritonbench_g.rbe_triton_transform.Correct
import bench.repairs.tritonbench_g.rbe_triton_transform.FPEquiv

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Audit
open VeriTile Triton
noncomputable section


def launchCounterexample : Torch.TensorMetadata :=
  ⟨.floating .fp32,.strided,⟨"cuda",0⟩,[1,131071,1],Bool.true,Bool.false,BitVec.ofNat 64 4096⟩

theorem admitted_grid_y_overflow :
    Contract.checkInputs launchCounterexample ⟨Bool.true,0⟩=.ok () ∧
    grid ⟨1,131071,1,0⟩=65536 ∧ ¬Index.SupportedLaunch ⟨1,131071,1,0⟩ := by
  refine ⟨by decide,by decide,?_⟩
  intro supported
  have := supported.program (by decide)
  norm_num [grid,rowParts,parts] at this

/-- At position zero, the stride-two logical pair is (1,3), whereas the
original unscaled imaginary-neighbor address reads backing element 1 = 2. -/
theorem original_stride_two_counterexample :
    Spec.result ⟨1,1,2,0⟩ (fun o => if o=0 then 1 else 3) 0 0 1=3 ∧
    Spec.result ⟨1,1,2,0⟩ (fun o => if o=0 then 1 else 2) 0 0 1=2 := by
  norm_num [Spec.result,Spec.pair,Spec.angle,Request.offset]

#guard_msgs (drop info) in
#specNonCircular Binding.Input avoiding [Selected.source,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#specNonCircular Binding.Output avoiding [Selected.source,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#specNonCircular Correct.idealResult avoiding [Selected.source,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#specNonCircular Bytes.nativeResult avoiding [Selected.source,Runtime.run,FP.Structural.exec]
#axiomsClean Correct.kernel_correct
#axiomsClean FPEquiv.kernel_native_equivalence
#axiomsClean Selected.carrier_types
#axiomsClean Selected.store_types
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Audit
