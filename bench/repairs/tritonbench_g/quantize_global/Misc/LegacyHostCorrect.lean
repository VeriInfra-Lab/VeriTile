import bench.repairs.tritonbench_g.quantize_global.Kernels
import bench.repairs.tritonbench_g.quantize_global.Contract
import bench.repairs.tritonbench_g.quantize_global.Proofs.Tensor
import bench.repairs.tritonbench_g.quantize_global.Proofs.Real

namespace QuantizeGlobalRepair.Direct.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- Independent complete target: one mathematical global maximum, nearest-even
integer quantization, both returned allocations and the exact exterior frame. -/
def Result (x : TensorMetadata) (X : RegionName) (alloc : Direct.Allocation x X)
    (initial : State ℝ) (ret : Direct.Returned ℝ) : Prop :=
  ret.output=alloc.output ∧ ret.maximum=alloc.maximum ∧
  ret.output.metadata.shape=x.shape ∧ ret.output.metadata.dtype=.other "int8" ∧
  ret.output.metadata.device=x.device ∧ ret.maximum.metadata.shape=[1] ∧
  ret.maximum.metadata.dtype=.floating .fp32 ∧ ret.maximum.metadata.device=x.device ∧
  ∃ m,Real.Maximum x.numel (inputValues X initial) m ∧
    ret.state.mem alloc.maximum.region 0=.mk .real m ∧
    (∀ i,i<x.numel → ret.state.mem alloc.output.region i=
      .mk .int (Real.quantized (inputValues X initial i) m)) ∧
    (∀ r i,¬(r=alloc.maximum.region ∧ i=0) → ¬(r=alloc.output.region ∧ i<x.numel) →
      ret.state.mem r i=initial.mem r i)

theorem python_checks_iff_preconditions (x : ArgumentMetadata) (finite : Bool) :
    checkInputs x finite=.ok () ↔ AdmittedArguments x finite := checkInputs_iff x finite

theorem quantize_global_correct (rest : Algebra ℝ) (x : TensorMetadata) (X : RegionName)
    (finite : Bool) (h : InputContract x finite) (alloc : Direct.Allocation x X) (tree : MaxTree)
    (covers : tree.Covers x.numel) (trials : Nat → List Config) (initial : State ℝ)
    (nonempty : ∀ k,k<segmentCount x.numel → trials k≠[]) :
    NativeLaunchSafe x.numel trials ∧
    ∃ ret,run rest Real.arithmetic abs max tree x X finite alloc trials initial=.ok (some ret) ∧
      Result x X alloc initial ret := by
  obtain ⟨ret,hr,ho,hm,hmem⟩ := host_execution rest Real.arithmetic abs max tree x X finite h alloc trials initial nonempty
  let xs := inputValues X initial
  let m := (hostTree x tree).eval 0 abs max xs
  have hM : Real.Maximum x.numel xs m := Real.tree_maximum _ x.numel xs (hostTree_covers x tree covers)
  change ret.state.mem=complete (parameters x X alloc) (fun i => scalar Real.arithmetic (xs i) m)
    (prepare x X alloc m initial).mem at hmem
  refine ⟨native_launch_safe x.numel trials h.extent,ret,hr,ho,hm,?_,?_,?_,?_,?_,?_,m,hM,?_,?_,?_⟩
  · rw [ho];exact alloc.outputShape
  · rw [ho];exact alloc.outputType
  · rw [ho];exact alloc.outputDevice
  · rw [hm];exact alloc.maximumShape
  · rw [hm];exact alloc.maximumType
  · rw [hm];exact alloc.maximumDevice
  · rw [hmem]
    simp [complete,parameters,prepare,alloc.separated.2.2]
  · intro i hi
    rw [hmem]
    simp [complete,parameters,hi,Real.signed_readback _ _ hM.1 (hM.2.1 i hi)]
    rfl
  · intro r i hm ho
    rw [hmem]
    exact full_frame x X alloc m _ initial r i hm ho


/-- Public original-Tensor binding: the complete result and every logical input
coordinate refer to the same actual storage, including nonzero storage offsets. -/
theorem quantize_global_tensor_correct (rest : Algebra ℝ)
    (x : InputTensor) (laws : InputTensorLaws x) (finite : Bool)
    (h : InputContract x.metadata finite) (read : Nat → Direct.Word)
    (interpret : Direct.Word → ℝ) (X : RegionName) (initial : State ℝ)
    (binding : TensorBinding x laws h read interpret X initial)
    (alloc : Direct.Allocation x.metadata X) (tree : MaxTree) (covers : tree.Covers x.metadata.numel)
    (trials : Nat → List Config) (nonempty : ∀ k,k<segmentCount x.metadata.numel → trials k≠[]) :
    NativeLaunchSafe x.metadata.numel trials ∧
    (∀ i,i<x.metadata.numel → inputValues X initial i=
      interpret (VeriTile.Torch.FP32.logicalValue read x laws i)) ∧
    ∃ ret,run rest Real.arithmetic abs max tree x.metadata X finite alloc trials initial=.ok (some ret) ∧
      Result x.metadata X alloc initial ret := by
  obtain ⟨safe,result⟩ := quantize_global_correct rest x.metadata X finite h alloc tree covers trials initial nonempty
  exact ⟨safe,original_logical_input x laws h read interpret X initial binding,result⟩

#axiomsClean quantize_global_tensor_correct

#specNonCircular Result avoiding [run,execute,executeProgram,source,kernel,scalar]
#axiomsClean quantize_global_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.Direct.Correct
