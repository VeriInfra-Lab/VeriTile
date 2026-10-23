import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Misc.RawCorrect
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Misc.RawNative

/-! Full original Tensor applications of the exposed raw kernel results. -/
namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.TensorBridge
open VeriTile Triton FusedRwkv6Kernel
open RawNative
noncomputable section
attribute [local instance] Classical.propDecidable

theorem rwkv_real (c : Shape) (x : Inputs) (raw : PythonNumber)
    (admitted : InputContract c x raw) (scale : ℝ) (m : Memory ℝ)
    (base : FusedRwkv6Kernel.Region → Nat) (buffers : Buffers c x base) :
    ∃ out,Source.launch Real.algebra c scale (Source.cells m)=some out ∧
      buffers.scratch.Valid ∧ buffers.scratch.metadata.shape=Checked.tensorShape c .o ∧
      buffers.scratch.metadata.device=(x.tensor .q).metadata.device ∧
      buffers.scratch.metadata.dataPtr.toNat=base .o ∧
      (∀ z<c.Z,∀ ik<c.NK,∀ t<c.T,∀ f<c.V,
        Source.words out .o (partialAddress c z ik t f)=
          Real.partialFormula c (inputs c m) scale z ik (visit c t) f) ∧
      (c.final=Bool.true → buffers.terminal.Valid ∧
        buffers.terminal.metadata.shape=Checked.tensorShape c .ht ∧
        buffers.terminal.metadata.device=(x.tensor .q).metadata.device ∧
        buffers.terminal.metadata.dataPtr.toNat=base .ht ∧
        ∀ z<c.Z,∀ e<c.K,∀ f<c.V,Source.words out .ht (stateAddress c z e f)=
          realState c (inputs c m) z e f c.T) ∧
      (∀ r b,¬Output c r b → Source.words out r b=m r b) := by
  have hc := (Checked.domain c x (base_checked c x raw admitted)).sizes
  obtain ⟨out,run,partials,terminal,frame⟩ := RawCorrect.rwkv_real c hc scale m
  refine ⟨out,run,buffers.scratchMetadata.1,buffers.scratchMetadata.2.1,
    buffers.scratchMetadata.2.2.1,buffers.scratchMetadata.2.2.2,partials,?_,frame⟩
  intro hs
  exact ⟨(buffers.terminalMetadata hs).1,(buffers.terminalMetadata hs).2.1,
    (buffers.terminalMetadata hs).2.2.1,(buffers.terminalMetadata hs).2.2.2,terminal hs⟩

theorem rwkv_native (a : Native.Instructions) (compiler : Native.Compiler)
    (p : Native.ReductionPlan) (binding : Native.ScalarBinding) (raw : PythonNumber)
    (c : Shape) (x : Inputs) (admitted : InputContract c x raw)
    (layout : Physical.Layout c) (buffers : Buffers c x layout.base)
    (direct : Checked.DirectInputs c x layout.base) (memory : Bytes) :
    ∃ result,execute a compiler p binding raw c layout memory=some result ∧
      buffers.scratch.Valid ∧ buffers.scratch.metadata.shape=Checked.tensorShape c .o ∧
      buffers.scratch.metadata.device=(x.tensor .q).metadata.device ∧
      (∀ z<c.Z,∀ ik<c.NK,∀ t<c.T,∀ f<c.V,
        tensorWord buffers.scratch result (partialAddress c z ik t f)=
          reference a compiler p binding raw c (Physical.loadWords c layout memory)
            .o (partialAddress c z ik t f)) ∧
      (c.final=Bool.true → buffers.terminal.Valid ∧
        buffers.terminal.metadata.shape=Checked.tensorShape c .ht ∧
        buffers.terminal.metadata.device=(x.tensor .q).metadata.device ∧
        ∀ z<c.Z,∀ e<c.K,∀ f<c.V,
          tensorWord buffers.terminal result (stateAddress c z e f)=
            reference a compiler p binding raw c (Physical.loadWords c layout memory)
              .ht (stateAddress c z e f)) ∧
      (∀ r∈Checked.inputRegions c,∀ b,
        tensorWord (x.tensor r) memory b=Physical.loadWords c layout memory r b) ∧
      (∀ r,InputRegion r → ∀ b,b<extent c r → ∀ j : Fin 4,
        result (layout.base r+4*b+j.val)=memory (layout.base r+4*b+j.val)) ∧
      (∀ addr,Physical.Outside c layout addr → result addr=memory addr) := by
  have hc := (Checked.domain c x (base_checked c x raw admitted)).sizes
  obtain ⟨result,run,words,inputFrame,frame⟩ :=
    RawNative.rwkv_native a compiler p binding raw c hc layout memory
  refine ⟨result,run,buffers.scratchMetadata.1,buffers.scratchMetadata.2.1,
    buffers.scratchMetadata.2.2.1,?_,?_,?_,inputFrame,frame⟩
  · intro z hz ik hik t ht f hf
    rw [tensor_word c layout buffers.scratch .o buffers.scratchMetadata.2.2.2]
    exact words .o _ (partial_output c z ik t f hz hik ht hf)
  · intro hs
    refine ⟨(buffers.terminalMetadata hs).1,(buffers.terminalMetadata hs).2.1,
      (buffers.terminalMetadata hs).2.2.1,?_⟩
    intro z hz e he f hf
    rw [tensor_word c layout buffers.terminal .ht (buffers.terminalMetadata hs).2.2.2]
    exact words .ht _ (terminal_output c z e f hs hz he hf)
  · intro r hr b
    exact tensor_word c layout (x.tensor r) r (direct.pointers r hr) memory b

#axiomsClean rwkv_real
#axiomsClean rwkv_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.TensorBridge
