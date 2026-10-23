import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Preparation
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.InitialReturned
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

/-- Local Python-float marshalling observations. `ratio` is the exact finite
Python scalar (as_integer_ratio), with a separate negative-zero observation.
Both explicit and default modes must marshal this same checked scalar. -/
structure ScaleABI (scalar : Contract.Scalar) (c : Shape) (ports : Numeric.ScalarPorts) where
  ratio : Rat
  negativeZero : Bool
  value_eq : scalar.value=(ratio.num:ℝ)/(ratio.den:ℝ)
  mode : Numeric.ScaleMode
  launch : Word
  marshal : launch=(VeriTile.Triton.FP.round .fp32 .ieeeValue ratio negativeZero).bits
  selected : launch=Numeric.selectScale ports.hostScale c mode

/-- The ordinary ctx.scale save/restore law, including the exact word consumed
by the later launcher. It cannot select a new backward numerical scale. -/
structure SavedScale (scalar : ScaleABI value c ports) where
  python : Contract.Scalar
  savedPython : python=value
  launch : Word
  restored : launch=scalar.launch

theorem selected_scale (models : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws models ports) (c : Shape) (mode : Numeric.ScaleMode) :
    Numeric.selectScale ports.hostScale c mode=Numeric.selectScale models.hostScale c mode := by
  cases mode <;> simp only [Numeric.selectScale,laws.hostScale]

theorem forward_optional_final_bytes (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports)
    (c : Shape) (storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed rf terminal ri) (state : σ)
    (enabled : rf=Bool.true) (present : storage.present .finalState=Bool.true) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V → ∀ lane,lane<4 →
      abi.bytes (forwardEntry abi ports c storage scale compiler state).2
        ((storage.tensor .finalState).metadata.dataPtr.toNat+4*c.stateOffset h e j+lane)=
      wordByte (Reference.finalState (wordOps (Numeric.referenceISA models)) c
        (inputsOfHeap (storage.binding.view 0 (abi.words state))) h e j) lane := by
  cases rf <;> simp only [Bool.false_eq_true] at enabled
  exact forward_final_bytes abi models ports laws c storage scale compiler state present

theorem backward_optional_initial_bytes (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports)
    (c : Shape) (storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed rf terminal ri) (checkpoint : Bool) (state : σ)
    (enabled : ri=Bool.true) (present : storage.present .initialGradient=Bool.true) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V → ∀ lane,lane<4 →
      abi.bytes (backwardEntry abi ports c storage scale compiler checkpoint state).2
        ((storage.tensor .initialGradient).metadata.dataPtr.toNat+4*c.stateOffset h e j+lane)=
      wordByte ((wordOps (Numeric.referenceISA models)).cast
        (Reference.reverseCarry (wordOps (Numeric.referenceISA models)) c
          (inputsOfHeap (storage.binding.view 0 (abi.words state))) scale h c.chunks e j)) lane := by
  cases ri <;> simp only [Bool.false_eq_true] at enabled
  exact backward_initial_bytes abi models ports laws c storage scale compiler checkpoint state present

/-- Native returned bytes refer to the actual original argument objects. The
only numerical premises are eight fixed-arity scalar laws. All source bytes
outside the exact scalar store footprint are preserved; full caller backings
are preserved across allocation and the source launch. -/
specification chunk_retention_ops_repaired_fp_returns
    (abi : MemoryABI σ) (models : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws models ports)
    (a : Contract.Forward) (admitted : Contract.ForwardContract a) (call : ForwardCall a admitted)
    (compiler : Compiler (shape a admitted) tree a.scale.value a.initial.isSome a.final terminal ri)
    (scalar : ScaleABI a.scale (shape a admitted) ports) (before prepared : σ)
    (preparation : PreparedInputs (shape a admitted) call.storage (0:Word)
      (abi.words before) (abi.words prepared) (forwardObjects a))
    (allocationFrame : ∀ t∈callerInputs a,∀ byte,byte<t.bytes →
      abi.bytes prepared (t.base+byte)=abi.bytes before (t.base+byte)) :
    let c := shape a admitted
    let scale := scalar.launch
    let targetScale := Numeric.selectScale models.hostScale c scalar.mode
    let result := forwardEntry abi ports c call.storage scale compiler prepared
    let xs := inputsOfHeap (originalHeap c (0:Word) (abi.words before) (forwardObjects a))
    (Contract.forwardCheck a=Bool.true ↔ Contract.ForwardContract a) ∧
    HostABI.LaunchABI c ∧ result.1=forwardTensors call.storage a.final ∧
    result.1.1.metadata.shape=Contract.valueShape a.dims ∧
    result.1.1.metadata.dtype=.floating .fp32 ∧ result.1.1.metadata.device=a.k.metadata.device ∧
    (∀ h t j,h<c.heads → t<c.T → j<c.V → ∀ lane,lane<4 →
      abi.bytes result.2 (result.1.1.metadata.dataPtr.toNat+4*c.valueOffset h t j+lane)=
      wordByte (Reference.output (wordOps (Numeric.referenceISA models)) c xs targetScale h t j) lane) ∧
    (a.final=Bool.true → ∀ h e j,h<c.heads → e<c.K → j<c.V → ∀ lane,lane<4 →
      abi.bytes result.2 ((call.storage.tensor .finalState).metadata.dataPtr.toNat+4*c.stateOffset h e j+lane)=
      wordByte (Reference.finalState (wordOps (Numeric.referenceISA models)) c xs h e j) lane) ∧
    (∀ t∈callerInputs a,∀ byte,byte<t.bytes → abi.bytes result.2 (t.base+byte)=abi.bytes before (t.base+byte)) ∧
    (∀ byte,(∀ s∈forwardCode c tree a.final,byte/4≠call.storage.binding.address s.address) →
      abi.bytes result.2 byte=abi.bytes prepared byte) := by
  dsimp only
  obtain ⟨hs,hd,hdev,_⟩ := returned_metadata call.storage .output call.output
  refine ⟨Contract.forwardCheck_iff a,Contract.launchABI _ admitted.1,rfl,hs,hd,hdev.trans call.device,?_,?_,?_,?_⟩
  · have result := forward_output_bytes abi models ports laws _ call.storage
      scalar.launch compiler prepared call.output
    rw [preparation.inputs,scalar.selected,selected_scale models ports laws (shape a admitted) scalar.mode] at result
    -- The host law changes both the native launch scalar and the reference
    -- scalar; restore the identical launch representation on the left.
    simpa only [scalar.selected,selected_scale models ports laws (shape a admitted) scalar.mode] using result
  · intro enabled
    have result := forward_optional_final_bytes abi models ports laws _ call.storage
      scalar.launch compiler prepared enabled (call.final enabled)
    rw [preparation.inputs] at result
    exact result
  · exact forward_backing_frame abi ports a admitted call _ compiler before prepared allocationFrame
  · intro byte outside
    change abi.bytes (nativeExecute abi ports _ call.storage.binding _ compiler.forward prepared) byte=_
    rw [compiler.forward_eq]
    exact native_byte_frame abi ports _ call.storage.binding _ _ prepared byte outside

/-- The later autograd call uses saved-Tensor versions and executes history
recomputation when checkpoint_level=1. References read original raw cotangents,
including strided/expanded inputs; their copies are justified elementwise. -/
specification chunk_retention_ops_repaired_fp_backward
    (abi : MemoryABI σ) (models : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws models ports)
    (a : Contract.Backward) (admitted : Contract.BackwardContract a)
    (fcall : ForwardCall a.forward admitted.1) (call : BackwardCall a admitted needsInitial)
    (compiler : Compiler (shape a.forward admitted.1) tree a.forward.scale.value
      a.forward.initial.isSome a.forward.final a.terminal.isSome (a.forward.initial.isSome && needsInitial))
    (scalar : ScaleABI a.forward.scale (shape a.forward admitted.1) ports) (savedScale : SavedScale scalar)
    (before fprepared original later : σ)
    (fpreparation : PreparedInputs (shape a.forward admitted.1) fcall.storage (0:Word)
      (abi.words before) (abi.words fprepared) (forwardObjects a.forward))
    (preparation : PreparedInputs (shape a.forward admitted.1) call.storage (0:Word)
      (abi.words original) (abi.words later) (backwardObjects a))
    (versions : Versions (shape a.forward admitted.1)
      (fcall.storage.binding.view 0 (abi.words (forwardEntry abi ports _ fcall.storage scalar.launch compiler fprepared).2))
      (call.storage.binding.view 0 (abi.words later)) (decide (a.forward.checkpoint=1)))
    (allocationFrame : ∀ t∈backwardCallers a,∀ byte,byte<t.bytes →
      abi.bytes later (t.base+byte)=abi.bytes original (t.base+byte)) :
    let c := shape a.forward admitted.1
    let result := backwardEntry abi ports c call.storage savedScale.launch compiler (decide (a.forward.checkpoint=1)) later
    let xs := Separate.joined (originalHeap c (0:Word) (abi.words before) (forwardObjects a.forward))
      (originalHeap c (0:Word) (abi.words original) (backwardObjects a))
    (Contract.backwardCheck a=Bool.true ↔ Contract.BackwardContract a) ∧
    HostABI.LaunchABI c ∧ result.1=backwardTensors call.storage (a.forward.initial.isSome && needsInitial) ∧
    (∀ r∈[Region.dq,Region.dk,Region.dv],
      (call.storage.tensor r).metadata.shape=regionShape c r ∧
      (call.storage.tensor r).metadata.dtype=.floating .fp32 ∧
      (call.storage.tensor r).metadata.device=a.forward.k.metadata.device ∧
      (call.storage.tensor r).metadata.isContiguous=Bool.true) ∧
    (∀ address,address.Valid c → address.region∈[Region.dq,Region.dk,Region.dv] → ∀ lane,lane<4 →
      abi.bytes result.2 ((call.storage.tensor address.region).metadata.dataPtr.toNat+4*address.offset c+lane)=
      wordByte (gradientTarget (wordOps (Numeric.referenceISA models)) c xs scalar.launch tree address) lane) ∧
    ((a.forward.initial.isSome && needsInitial)=Bool.true → ∀ h e j,h<c.heads → e<c.K → j<c.V →
      ∀ lane,lane<4 → abi.bytes result.2
        ((call.storage.tensor .initialGradient).metadata.dataPtr.toNat+4*c.stateOffset h e j+lane)=
      wordByte ((wordOps (Numeric.referenceISA models)).cast
        (Reference.reverseCarry (wordOps (Numeric.referenceISA models)) c
          (inputsOfHeap (originalHeap c (0:Word) (abi.words original) (backwardObjects a))) scalar.launch h c.chunks e j)) lane) ∧
    (∀ t∈backwardCallers a,∀ byte,byte<t.bytes → abi.bytes result.2 (t.base+byte)=abi.bytes original (t.base+byte)) ∧
    (∀ byte,(∀ s∈recomputeCode c tree (decide (a.forward.checkpoint=1))++
        backwardCode c tree (a.forward.initial.isSome && needsInitial),byte/4≠call.storage.binding.address s.address) →
      abi.bytes result.2 byte=abi.bytes later byte) := by
  dsimp only
  rw [savedScale.restored]
  refine ⟨Contract.backwardCheck_iff a,Contract.launchABI _ admitted.1.1,rfl,?_,?_,?_,?_,?_⟩
  · intro r hr
    have present := call.outputs r (by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hr ⊢
      tauto)
    obtain ⟨hs,hd,hdev,hc⟩ := returned_metadata call.storage r present
    exact ⟨hs,hd,hdev.trans call.device,hc⟩
  · intro address valid tag lane hl
    have present := call.outputs address.region (by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at tag ⊢
      tauto)
    have result := backward_gradient_bytes abi models ports laws _ fcall.storage call.storage scalar.launch compiler
      (decide (a.forward.checkpoint=1)) fprepared later versions address valid tag present lane hl
    simpa only [Separate.joined,fpreparation.inputs,preparation.inputs] using result
  · intro enabled
    have result := backward_optional_initial_bytes abi models ports laws _ call.storage scalar.launch compiler
      (decide (a.forward.checkpoint=1)) later enabled (call.toBackwardAllocation.initial enabled)
    rw [preparation.inputs] at result
    exact result
  · exact backward_backing_frame abi ports a admitted call scalar.launch compiler original later allocationFrame
  · intro byte outside
    change abi.bytes (nativeExecute abi ports _ call.storage.binding _ _ later) byte=_
    rw [compiler.backward_eq]
    exact native_byte_frame abi ports _ call.storage.binding _ _ later byte outside

#axiomsClean chunk_retention_ops_repaired_fp_returns
#axiomsClean chunk_retention_ops_repaired_fp_backward
#specNonCircular MemoryABI avoiding [forwardEntry,backwardEntry,Reference.output,Reference.dq]
#specNonCircular Compiler avoiding [forwardEntry,backwardEntry]
#specNonCircular ScaleABI avoiding [forwardEntry,backwardEntry,Reference.output,Reference.dq]
#specNonCircular SavedScale avoiding [forwardEntry,backwardEntry,Reference.output,Reference.dq]
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
