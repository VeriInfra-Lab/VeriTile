import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Storage
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section

/-- Exact raw-word public result, complete byte frame, and the optional terminal
state. This theorem has no whole-output numerical assumption. -/
specification native_correct (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (hc : Domain c) (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) :
    (∀ z, z<c.Z → ∀ t, t<c.T → ∀ f, f<c.V →
      loadWords c layout (physicalForward a compiler c plan scale layout memory)
        .result (resultAddress c z t f)=
      Native.observe a compiler (scalarOutput Native.sourceOps c plan
        (inputs c (Native.entry (loadWords c layout memory))) (.word .outside 0 scale) z t f)) ∧
    (c.final=Bool.true → ∀ z, z<c.Z → ∀ e, e<c.K → ∀ f, f<c.V →
      loadWords c layout (physicalForward a compiler c plan scale layout memory)
        .ht (stateAddress c z e f)=
      Native.observe a compiler (.store32 (scalarState Native.sourceOps c
        (inputs c (Native.entry (loadWords c layout memory))) z e f c.T))) ∧
    (∀ addr, (∀ r b j, layout.decode addr=some (r,b,j) → ¬WholeOutput c r b) →
      physicalForward a compiler c plan scale layout memory addr=memory addr) := by
  refine ⟨?_,?_,?_⟩
  · intro z hz t ht f hf
    exact physical_returned a compiler c hc.sizes plan scale layout memory z t f hz ht hf
  · intro hs z hz e he f hf
    exact physical_terminal a compiler c hc.sizes plan scale layout memory hs z e f hz he hf
  · exact physical_frame a compiler c plan scale layout memory

/-- Native equivalence closes from four local opcode laws only. -/
theorem fpequiv (hardware reference : Native.Instructions)
    (h : Native.InstructionAgreement hardware reference) (compiler : Native.Compiler)
    (c : Shape) (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) :
    physicalForward hardware compiler c plan scale layout memory=
      physicalForward reference compiler c plan scale layout memory := by
  unfold physicalForward
  rw [Native.fpequiv hardware reference h]

specification native_equiv_correct (hardware reference : Native.Instructions)
    (h : Native.InstructionAgreement hardware reference) (compiler : Native.Compiler) (c : Shape)
    (hc : Domain c) (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) :
    (∀ z, z<c.Z → ∀ t, t<c.T → ∀ f, f<c.V →
      loadWords c layout (physicalForward hardware compiler c plan scale layout memory)
        .result (resultAddress c z t f)=
      Native.observe reference compiler (scalarOutput Native.sourceOps c plan
        (inputs c (Native.entry (loadWords c layout memory))) (.word .outside 0 scale) z t f)) ∧
    (c.final=Bool.true → ∀ z, z<c.Z → ∀ e, e<c.K → ∀ f, f<c.V →
      loadWords c layout (physicalForward hardware compiler c plan scale layout memory)
        .ht (stateAddress c z e f)=
      Native.observe reference compiler (.store32 (scalarState Native.sourceOps c
        (inputs c (Native.entry (loadWords c layout memory))) z e f c.T))) ∧
    (∀ addr, (∀ r b j, layout.decode addr=some (r,b,j) → ¬WholeOutput c r b) →
      physicalForward hardware compiler c plan scale layout memory addr=memory addr) := by
  rw [fpequiv hardware reference h compiler c plan scale layout memory]
  exact native_correct reference compiler c hc plan scale layout memory

#axiomsClean native_correct
#axiomsClean native_equiv_correct
#axiomsClean fpequiv
#specNonCircular scalarOutput avoiding [runLoop,runProgram,runGrid,forward,physicalForward]
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
