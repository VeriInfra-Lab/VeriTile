import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.KernelRealResults
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.Facts

namespace VeriTile.Bench.Repairs.RmsnormTriton.TensorBridge
open VeriTile
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section

/-- Project only address and launch data from an existing parameter record. -/
def geometry (p : Parameters) : KernelGrid.Geometry :=
  ⟨p.batches,p.rows,p.width,⟨p.input.allocation,p.input.base,p.input.stride⟩,
    ⟨p.weights.allocation,p.weights.base,p.weights.stride⟩,
    ⟨p.output.allocation,p.output.base,p.output.stride⟩,p.eps⟩

theorem source_geometry (p : Parameters) :
    TensorSource.source (geometry p).parameters=TensorSource.source p := rfl

/-- Successful complete typed source and candidate grids under the Real interpretation.
The theorem binds original Tensor values, every output, full physical memory,
its exterior frame, and the checked signed-index/pointer correspondence.
No allocation or launch-success premise is assumed. -/
theorem kernel_correct
    (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (red : Reduction) (p : Parameters) (before : Memory ℝ)
    (api : MemoryAPI p live) (binding : Binding x w out eps p api.byteBase)
    (views : TensorBinding x w out p api.byteBase) (storage : StorageBinding api views)
    (checked : x.numel≠0 → LaunchOutput x out) (outputDevice : out.device=x.device)
    (undefinedWeight : Nat → ℝ)
    (schedule : Schedule p) :
    Spec.Real (checkTensorInput x w eps=.ok () ∧
      (x.numel≠0 → checkOutput x out=.ok ()) ∧
      MachineResult x w out eps c p api binding checked ∧ ActualAddresses p views ∧ ActualStorageSafe p views ∧
      views.output.metadata.shape=views.input.metadata.shape ∧
      views.output.metadata.device=views.input.metadata.device ∧
      views.output.metadata.dtype=.floating .fp32 ∧
      ∃ final, TensorSource.grid p schedule.order before=some final ∧
        TensorSource.candidateGrid p schedule.order before=some final ∧
        kernelRun realNumerics red p undefinedWeight schedule before=some final ∧
        (∀ b m k,b<p.batches → m<p.rows → k<p.width →
          tensorRead final views.output (flatIndex p b m k)=
            mathematicalResult p views.input views.weight before b m k) ∧
        final=referenceMemory p (mathematicalResult p views.input views.weight before) before ∧
        (∀ a i,Outside p a i → final a i=before a i)) := by
  let fresh := api.separation
  let unique := api.outputUnique
  refine ⟨(checkTensorInput_iff _ _ _).2 c,fun h => (checkOutput_iff _ _).2 (checked h),
    machine_result x w out eps c p api binding checked,
    actual_addresses x w out eps c p api.byteBase binding views,actual_storage_safe api views storage,
    ?_,?_,views.outputLaws.dtype,?_⟩
  · rw [views.outputMetadata,views.inputMetadata]
    exact binding.outputShape
  · rw [views.outputMetadata,views.inputMetadata]
    exact outputDevice
  · obtain ⟨final,source,candidate,ran,rawValues,_,frame⟩ :=
      KernelGrid.real red p fresh unique undefinedWeight schedule before
    have values : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
        final p.output.allocation (outputOffset p b m k)=
          mathematicalResult p views.input views.weight before b m k := by
      intro b m k hb hm hk
      apply (rawValues b m k hb hm hk).trans
      exact TensorSource.mathematical_congr p.width p.eps _ _ _ _ k hk
        (fun j hj => (views.input_read before b m j hb hm hj).symm)
        (fun j hj => (views.weight_read before j hj).symm)
    exact ⟨final,source,candidate,ran,fun b m k hb hm hk =>
      (views.output_read final b m k hb hm hk).trans (values b m k hb hm hk),
      memory_ext p _ before final values frame,frame⟩

/-- Actual typed source and candidate grids execute successfully. Their complete
symbolic heap is observed with explicit native shape/operation choices and
scalar FP32 primitives; every actual Tensor output and the whole physical heap
agree with the independent logical recurrence. No floating identity is assumed. -/
theorem kernel_native_equivalence
    (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (Q : Numerics Torch.FP32.Word) (profile : Symbolic.Profile)
    (p : Parameters) (before : Memory Torch.FP32.Word)
    (api : MemoryAPI p live) (binding : Binding x w out eps p api.byteBase)
    (views : TensorBinding x w out p api.byteBase) (storage : StorageBinding api views)
    (checked : x.numel≠0 → LaunchOutput x out) (outputDevice : out.device=x.device)
    (undefinedWeight : Nat → Torch.FP32.Word)
    (schedule : Schedule p) :
    checkTensorInput x w eps=.ok () ∧
      (x.numel≠0 → checkOutput x out=.ok ()) ∧
      MachineResult x w out eps c p api binding checked ∧ ActualAddresses p views ∧ ActualStorageSafe p views ∧
      views.output.metadata.shape=views.input.metadata.shape ∧
      views.output.metadata.device=views.input.metadata.device ∧
      views.output.metadata.dtype=.floating .fp32 ∧
      ∃ symbolicFinal final,
        Symbolic.grid p (fun k=>.input (undefinedWeight k)) schedule.order (Symbolic.entryGraph before)=some symbolicFinal ∧
        Symbolic.candidateGrid p (fun k=>.input (undefinedWeight k)) schedule.order (Symbolic.entryGraph before)=some symbolicFinal ∧
        Symbolic.observedGrid Q profile p undefinedWeight schedule.order before=some final ∧
        Symbolic.observedCandidateGrid Q profile p undefinedWeight schedule.order before=some final ∧
        final=Symbolic.projectMemory Q profile symbolicFinal ∧
        (∀ b m k,b<p.batches → m<p.rows → k<p.width →
          tensorRead final views.output (flatIndex p b m k)=
            Symbolic.nativeTensorResult Q profile p views.input views.weight before b m k) ∧
        final=referenceMemory p (Symbolic.nativeTensorResult Q profile p views.input views.weight before) before ∧
        (∀ a i,Outside p a i → final a i=before a i) := by
  refine ⟨(checkTensorInput_iff _ _ _).2 c,fun h => (checkOutput_iff _ _).2 (checked h),
    machine_result x w out eps c p api binding checked,
    actual_addresses x w out eps c p api.byteBase binding views,actual_storage_safe api views storage,
    ?_,?_,views.outputLaws.dtype,?_⟩
  · rw [views.outputMetadata,views.inputMetadata]
    exact binding.outputShape
  · rw [views.outputMetadata,views.inputMetadata]
    exact outputDevice
  · obtain ⟨symbolicFinal,final,source,candidate,observed,observedCandidate,projected,rawValues,_,frame⟩ :=
      KernelGrid.native Q profile p api.separation api.outputUnique undefinedWeight schedule before
    have values : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
        final p.output.allocation (outputOffset p b m k)=
          Symbolic.nativeTensorResult Q profile p views.input views.weight before b m k := by
      intro b m k hb hm hk
      apply (rawValues b m k hb hm hk).trans
      apply congrArg (Symbolic.observe Q profile)
      exact Symbolic.logical_congr p.width blockSize p.eps _ _ _ _
        (fun j hj => congrArg Symbolic.Term.input (views.input_read before b m j hb hm hj).symm)
        (fun j hj => congrArg Symbolic.Term.input (views.weight_read before j hj).symm) k hk
    exact ⟨symbolicFinal,final,source,candidate,observed,observedCandidate,projected,
      fun b m k hb hm hk => (views.output_read final b m k hb hm hk).trans (values b m k hb hm hk),
      memory_ext p _ before final values frame,frame⟩

#axiomsClean kernel_correct
#axiomsClean kernel_native_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormTriton.TensorBridge
