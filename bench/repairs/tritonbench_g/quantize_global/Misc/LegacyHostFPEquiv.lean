import bench.repairs.tritonbench_g.quantize_global.Kernels
import bench.repairs.tritonbench_g.quantize_global.Contract
import bench.repairs.tritonbench_g.quantize_global.Proofs.Tensor
import bench.repairs.tritonbench_g.quantize_global.Proofs.FP

namespace QuantizeGlobalRepair.Direct.FPEquiv
open VeriTile.Triton FP.Structural
open Floating (Word)
noncomputable section

/-- The independent FP32 profile uses RNE division, a separate multiplication,
explicit clamp comparisons, nearest-even signed64 conversion and int8 narrowing. -/
def Result (x : TensorMetadata) (X : RegionName) (alloc : Direct.Allocation x X)
    (tree : MaxTree) (initial : State Word) (ret : Direct.Returned Word) : Prop :=
  ret.output=alloc.output ∧ ret.maximum=alloc.maximum ∧
  ret.output.metadata.shape=x.shape ∧ ret.output.metadata.dtype=.other "int8" ∧
  ret.output.metadata.device=x.device ∧ ret.maximum.metadata.shape=[1] ∧
  ret.maximum.metadata.dtype=.floating .fp32 ∧ ret.maximum.metadata.device=x.device ∧
  let m := (hostTree x tree).eval Floating.zero FP.Value.abs Floating.maximum (inputValues X initial)
  ret.state.mem alloc.maximum.region 0=.mk .real m ∧
  (∀ i,i<x.numel → ret.state.mem alloc.output.region i=
    .mk .int (Floating.output FP.Config.ieeeValue (inputValues X initial i) m).toInt) ∧
  (∀ r i,¬(r=alloc.maximum.region ∧ i=0) → ¬(r=alloc.output.region ∧ i<x.numel) →
    ret.state.mem r i=initial.mem r i)

/-- No numerical law is needed for complete source-order execution and frame. -/
theorem quantize_global_source_fp32 (rest : Algebra Word) (a : Scalar Word)
    (absolute : Word → Word) (maxOp : Word → Word → Word) (tree : MaxTree)
    (x : TensorMetadata) (X : RegionName) (finite : Bool) (h : InputContract x finite)
    (alloc : Direct.Allocation x X) (trials : Nat → List Config) (initial : State Word)
    (nonempty : ∀ k,k<segmentCount x.numel → trials k≠[]) :
    NativeLaunchSafe x.numel trials ∧
    ∃ ret,run rest a absolute maxOp tree x X finite alloc trials initial=.ok (some ret) ∧
      ret.output=alloc.output ∧ ret.maximum=alloc.maximum ∧
      let xs := inputValues X initial
      let m := (hostTree x tree).eval (a.literal 0) absolute maxOp xs
      ret.state.mem=complete (parameters x X alloc) (fun i => scalar a (xs i) m)
        (prepare x X alloc m initial).mem :=
  ⟨native_launch_safe x.numel trials h.extent,
    host_execution rest a absolute maxOp tree x X finite h alloc trials initial nonempty⟩

/-- Optional prior primitive-conformance theorem. These model-comparison
premises are not deferred task-level numerical transformation experiments.
No output, index, mask, coverage, allocation or frame equality is assumed. -/
theorem quantize_global_fp32_if_local (rest : Algebra Word) (a : Scalar Word)
    (absolute : Word → Word) (maxOp : Word → Word → Word) (tree : MaxTree)
    (x : TensorMetadata) (X : RegionName) (finite : Bool) (h : InputContract x finite)
    (alloc : Direct.Allocation x X) (trials : Nat → List Config) (initial : State Word)
    (nonempty : ∀ k,k<segmentCount x.numel → trials k≠[])
    (zeroPort : a.literal 0=Floating.zero)
    (hostPorts : Floating.HostPorts absolute maxOp (inputValues X initial) (hostTree x tree))
    (ports : ∀ i,i<x.numel → Floating.LocalPorts a FP.Config.ieeeValue (inputValues X initial i)
      ((hostTree x tree).eval Floating.zero FP.Value.abs Floating.maximum (inputValues X initial))) :
    ∃ ret,run rest a absolute maxOp tree x X finite alloc trials initial=.ok (some ret) ∧
      Result x X alloc tree initial ret := by
  obtain ⟨ret,hr,ho,hm,hmem⟩ := host_execution rest a absolute maxOp tree x X finite h alloc trials initial nonempty
  let xs := inputValues X initial
  let m := (hostTree x tree).eval Floating.zero FP.Value.abs Floating.maximum xs
  have maxEq : (hostTree x tree).eval (a.literal 0) absolute maxOp xs=m := by
    rw [zeroPort]
    exact Floating.maximum_equivalent absolute maxOp xs _ hostPorts
  dsimp only at hmem
  rw [maxEq] at hmem
  refine ⟨ret,hr,ho,hm,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · rw [ho];exact alloc.outputShape
  · rw [ho];exact alloc.outputType
  · rw [ho];exact alloc.outputDevice
  · rw [hm];exact alloc.maximumShape
  · rw [hm];exact alloc.maximumType
  · rw [hm];exact alloc.maximumDevice
  · rw [hmem]
    simp [complete,parameters,prepare,alloc.separated.2.2,m,xs]
  · intro i hi
    rw [hmem]
    simp [complete,parameters,hi,m,xs,Floating.scalar_equivalent a FP.Config.ieeeValue _ _ (ports i hi)]
  · intro r i hm ho
    rw [hmem]
    exact full_frame x X alloc m _ initial r i hm ho


/-- Retained optional host theorem with the original input binding and explicit
primitive-conformance premises; it is outside current kernel acceptance. -/
theorem quantize_global_tensor_fp32_if_local (rest : Algebra Word) (a : Scalar Word)
    (absolute : Word → Word) (maxOp : Word → Word → Word) (tree : MaxTree)
    (x : InputTensor) (laws : InputTensorLaws x) (finite : Bool) (h : InputContract x.metadata finite)
    (read : Nat → Word) (X : RegionName) (initial : State Word)
    (binding : TensorBinding x laws h read id X initial)
    (alloc : Direct.Allocation x.metadata X) (trials : Nat → List Config)
    (nonempty : ∀ k,k<segmentCount x.metadata.numel → trials k≠[])
    (zeroPort : a.literal 0=Floating.zero)
    (hostPorts : Floating.HostPorts absolute maxOp (inputValues X initial) (hostTree x.metadata tree))
    (ports : ∀ i,i<x.metadata.numel → Floating.LocalPorts a FP.Config.ieeeValue (inputValues X initial i)
      ((hostTree x.metadata tree).eval Floating.zero FP.Value.abs Floating.maximum (inputValues X initial))) :
    (∀ i,i<x.metadata.numel → inputValues X initial i=VeriTile.Torch.FP32.logicalValue read x laws i) ∧
    ∃ ret,run rest a absolute maxOp tree x.metadata X finite alloc trials initial=.ok (some ret) ∧
      Result x.metadata X alloc tree initial ret :=
  ⟨original_logical_input x laws h read id X initial binding,
    quantize_global_fp32_if_local rest a absolute maxOp tree x.metadata X finite h alloc trials initial
      nonempty zeroPort hostPorts ports⟩

#axiomsClean quantize_global_tensor_fp32_if_local

#specNonCircular Result avoiding [run,execute,executeProgram,source,kernel,scalar]
#axiomsClean quantize_global_source_fp32
#axiomsClean quantize_global_fp32_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.Direct.FPEquiv
