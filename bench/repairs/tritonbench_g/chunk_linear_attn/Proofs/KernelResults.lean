import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelPhysical

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelResults
open VeriTile Triton FP.Structural
open KernelPhysical KernelTensor
noncomputable section
set_option maxHeartbeats 10000000

theorem real_readback (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (logical : Memory ℝ) (before : Nat → Cell ℝ) (r : RegionName) (dst : Host.Tensor)
    (descriptor : Host.descriptor c a r=some dst) (output : Host.buffer r dst∈Host.outputBuffers a)
    (i : Nat) (hi : i<dst.shape.prod) :
    logicalReal (Host.publish a logical before) (resources.tensors.actual dst) i=(logical r i).read .real := by
  rw [KernelTensor.real_read dst _ (resources.tensors.binding r dst descriptor) _ i hi]
  exact congrArg (fun cell : Cell ℝ => cell.read .real)
    (Host.published_read c a resources.physical logical before r dst output i hi)

theorem native_readback (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (logical : RegionName → Nat → BitVec 32) (before : BM.Read) (r : RegionName) (dst : Host.Tensor)
    (descriptor : Host.descriptor c a r=some dst) (output : Host.buffer r dst∈Host.outputBuffers a)
    (i : Nat) (hi : i<dst.shape.prod) :
    logicalNative (publishBytes a logical before) (resources.tensors.actual dst)
      (resources.tensors.binding r dst descriptor).laws i=⟨logical r i⟩ := by
  rw [native_read dst _ (resources.tensors.binding r dst descriptor) _ i hi]
  congr 1
  exact bytes_read c a resources logical before (Host.buffer r dst) output i hi

/-- Full causal tensor formula and VJP on the actual shared output Tensors.
The input readback field connects every valid source input cell to its actual
Tensor storage interpretation. No source execution appears in this predicate. -/
structure RealTensorResult (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (before : Nat → Cell ℝ) (fallback : Memory ℝ) (after : Nat → Cell ℝ) : Prop where
  output : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (v : Fin c.dims.V),
    logicalReal after (resources.tensors.actual a.output) ((h.val*c.dims.T+t.val)*c.dims.V+v.val)=
      Real.output c.scale.scalarValue (Real.inputs c.mathCall.forward h.val (realMemory c a before fallback) c.initial.isSome) t v
  query : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (k : Fin c.dims.K),
    logicalReal after (resources.tensors.actual a.queryGradient) ((h.val*c.dims.T+t.val)*c.dims.K+k.val)=
      (c.mathCall.target h.val (realMemory c a before fallback)).q t k
  key : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (k : Fin c.dims.K),
    logicalReal after (resources.tensors.actual a.keyGradient) ((h.val*c.dims.T+t.val)*c.dims.K+k.val)=
      (c.mathCall.target h.val (realMemory c a before fallback)).k t k
  value : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (v : Fin c.dims.V),
    logicalReal after (resources.tensors.actual a.valueGradient) ((h.val*c.dims.T+t.val)*c.dims.V+v.val)=
      (c.mathCall.target h.val (realMemory c a before fallback)).v t v
  final : ∀ dst,a.finalState=some dst → ∀ (h : Fin (c.dims.B*c.dims.H)) (k : Fin c.dims.K) (v : Fin c.dims.V),
    logicalReal after (resources.tensors.actual dst) ((h.val*c.dims.K+k.val)*c.dims.V+v.val)=
      Real.finalState (Real.inputs c.mathCall.forward h.val (realMemory c a before fallback) c.initial.isSome) k v
  initial : ∀ dst,a.initialGradient=some dst → ∀ (h : Fin (c.dims.B*c.dims.H)) (k : Fin c.dims.K) (v : Fin c.dims.V),
    logicalReal after (resources.tensors.actual dst) ((h.val*c.dims.K+k.val)*c.dims.V+v.val)=
      (c.mathCall.target h.val (realMemory c a before fallback)).initial k v
  vjp : ∀ h : Fin (c.dims.B*c.dims.H),Real.IsVJP c.scale.scalarValue
    (Real.inputs c.mathCall.backward h.val (realMemory c a before fallback) c.initial.isSome)
    (fun t v => Real.incoming c.mathCall.backward h.val (realMemory c a before fallback) t.val v.val)
    (fun k v => Real.endpoint c.mathCall.backward h.val (realMemory c a before fallback) Bool.true c.terminal.isSome k.val v.val)
    (c.mathCall.target h.val (realMemory c a before fallback))
  inputReadback : ∀ r dst (_descriptor : Host.descriptor c a r=some dst), ∀ i,i<dst.shape.prod →
    (realMemory c a before fallback r i).read .real=logicalReal before (resources.tensors.actual dst) i
  inputs : ∀ r∈kernelInputRegions,∀ dst,Host.descriptor c a r=some dst → ∀ address,
    (resources.tensors.actual dst).storage.base≤4*address →
    4*address<(resources.tensors.actual dst).storage.base+4*(resources.tensors.actual dst).storage.elements →
    after address=before address
  frame : ∀ address,(∀ b∈Host.outputBuffers a,¬Physical.Contains b address) → after address=before address

theorem real_tensor (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (valid : Host.Valid c.dims) (before : Nat → Cell ℝ) (fallback : Memory ℝ) :
    RealTensorResult c a resources before fallback
      (Host.publish a (c.mathCall.result (realMemory c a before fallback)) before) := by
  let logical := c.mathCall.result (realMemory c a before fallback)
  have result := KernelExecution.real_model c.mathCall valid (realMemory c a before fallback)
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro h t v
    have bound : (h.val*c.dims.T+t.val)*c.dims.V+v.val<a.output.shape.prod := by
      rw [resources.layout.outputShape]; exact Host.packed_value c h.val h.isLt t v
    rw [real_readback c a resources logical before "output" a.output (by simp [Host.descriptor])
      (Host.output_mem a) _ bound]
    exact congrArg (fun x : Cell ℝ => x.read .real) (result.output h t v)
  · intro h t k
    have bound : (h.val*c.dims.T+t.val)*c.dims.K+k.val<a.queryGradient.shape.prod := by
      rw [resources.layout.queryShape]; exact Host.packed_query c h.val h.isLt t k
    rw [real_readback c a resources logical before "queryGradient" a.queryGradient (by simp [Host.descriptor])
      (Host.query_mem a) _ bound]
    exact congrArg (fun x : Cell ℝ => x.read .real) (result.query h t k)
  · intro h t k
    have bound : (h.val*c.dims.T+t.val)*c.dims.K+k.val<a.keyGradient.shape.prod := by
      rw [resources.layout.keyShape]; exact Host.packed_query c h.val h.isLt t k
    rw [real_readback c a resources logical before "keyGradient" a.keyGradient (by simp [Host.descriptor])
      (Host.key_mem a) _ bound]
    exact congrArg (fun x : Cell ℝ => x.read .real) (result.key h t k)
  · intro h t v
    have bound : (h.val*c.dims.T+t.val)*c.dims.V+v.val<a.valueGradient.shape.prod := by
      rw [resources.layout.valueShape]; exact Host.packed_value c h.val h.isLt t v
    rw [real_readback c a resources logical before "valueGradient" a.valueGradient (by simp [Host.descriptor])
      (Host.value_mem a) _ bound]
    exact congrArg (fun x : Cell ℝ => x.read .real) (result.value h t v)
  · intro dst hd h k v
    have shape := resources.layout.stateShapes dst (by simp [hd])
    have bound : (h.val*c.dims.K+k.val)*c.dims.V+v.val<dst.shape.prod := by
      rw [shape]; exact Host.packed_state c h.val h.isLt k v
    have flag : c.mathCall.final=Bool.true := by
      have hf := resources.layout.finalFlag
      simpa only [hd,Option.isSome_some] using hf.symm
    rw [real_readback c a resources logical before "final" dst (by simp [Host.descriptor,hd])
      (Host.final_mem a dst hd) _ bound]
    exact congrArg (fun x : Cell ℝ => x.read .real) (result.final flag h k v)
  · intro dst hd h k v
    have shape := resources.layout.stateShapes dst (by simp [hd])
    have bound : (h.val*c.dims.K+k.val)*c.dims.V+v.val<dst.shape.prod := by
      rw [shape]; exact Host.packed_state c h.val h.isLt k v
    have flag : c.mathCall.seed=Bool.true := by
      have hf := resources.layout.initialFlag
      simpa only [hd,Option.isSome_some] using hf.symm
    rw [real_readback c a resources logical before "initialGradient" dst (by simp [Host.descriptor,hd])
      (Host.initial_mem a dst hd) _ bound]
    exact congrArg (fun x : Cell ℝ => x.read .real) (result.initial flag h k v)
  · exact result.vjp
  · exact real_input c a resources.tensors before fallback
  · intro r hr dst hd address hlo hhi
    exact KernelPhysical.input_real c a resources logical before r hr dst hd address ⟨hlo,hhi⟩
  · exact Host.published_frame a logical before

structure NativeTensorResult (profile : NativeTerms.FP32Profile) (c : Host.Invocation)
    (a : Host.Allocations) (resources : Resources c a) (valid : Host.Valid c.dims)
    (before : BM.Read) (fallback : RegionName → Nat → BitVec 32) (after : BM.Read) : Prop where
  outputs : ∀ r dst (descriptor : Host.descriptor c a r=some dst), Host.buffer r dst∈Host.outputBuffers a →
    ∀ i,i<dst.shape.prod → logicalNative after (resources.tensors.actual dst)
      (resources.tensors.binding r dst descriptor).laws i=
      ⟨KernelExecution.nativeTarget profile c.mathCall valid c.scale.argumentFP32 (nativeMemory c a before fallback) r i⟩
  outputBytes : ∀ b∈Host.outputBuffers a,∀ i,i<b.words → ∀ j,j<4 → after (4*b.base+4*i+j)=
    BM.byte (KernelExecution.nativeTarget profile c.mathCall valid c.scale.argumentFP32
      (nativeMemory c a before fallback) b.region i) j
  inputReadback : ∀ r dst (descriptor : Host.descriptor c a r=some dst), ∀ i,i<dst.shape.prod →
    nativeMemory c a before fallback r i=
      (logicalNative before (resources.tensors.actual dst) (resources.tensors.binding r dst descriptor).laws i).bits
  inputs : ∀ r∈kernelInputRegions,∀ dst,Host.descriptor c a r=some dst → ∀ address,
    (resources.tensors.actual dst).storage.base≤address →
    address<(resources.tensors.actual dst).storage.base+4*(resources.tensors.actual dst).storage.elements →
    after address=before address
  frame : ∀ address,(∀ b∈Host.outputBuffers a,BM.Outside (view b) address) → after address=before address

theorem native_tensor (profile : NativeTerms.FP32Profile) (c : Host.Invocation)
    (a : Host.Allocations) (resources : Resources c a) (valid : Host.Valid c.dims)
    (before : BM.Read) (fallback : RegionName → Nat → BitVec 32) :
    NativeTensorResult profile c a resources valid before fallback (publishBytes a
      (KernelExecution.nativeTarget profile c.mathCall valid c.scale.argumentFP32 (nativeMemory c a before fallback)) before) := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro r dst hd ho i hi
    exact native_readback c a resources _ before r dst hd ho i hi
  · intro b hb i hi j hj
    exact bytes_at c a resources _ before b hb i j hi hj
  · exact native_input c a resources.tensors before fallback
  · intro r hr dst hd address hlo hhi
    exact input_bytes c a resources _ before r hr dst hd address ⟨hlo,hhi⟩
  · exact bytes_frame a _ before

#axiomsClean real_tensor
#axiomsClean native_tensor
#specNonCircular RealTensorResult avoiding [runReal,runNative,Host.execute,KernelExecution.run,
  KernelExecution.runNative,Real.Call.result,gridMemory,completeMemory]
#specNonCircular NativeTensorResult avoiding [runReal,runNative,Host.execute,KernelExecution.run,
  KernelExecution.runNative,Real.Call.result,gridMemory,completeMemory,NativeBound.rawMemory]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelResults
