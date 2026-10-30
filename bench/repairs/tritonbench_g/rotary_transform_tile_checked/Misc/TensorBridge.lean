import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.KernelIndex

/-! Optional Tensor byte observations and checked metadata/index bindings.
The complete Real/native kernel endpoints remain in `Correct` and `NativeCorrect`. -/
namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.TensorBridge
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
open KernelOutput KernelPublic
noncomputable section

/-- The existing local codec derives all four bytes from a proven output word. -/
theorem rotary_transform_output_bytes {M : Type}
    (api : Binding.API M Native.Word Physical.Byte) (codec : Binding.Codec api)
    (c : Checked.Call) (out : Binding.Output c) (tensor : VeriTile.Torch.TensorMetadata)
    (callerDevice : Nat) (cfg : Config) (m : M) (target : Nat→Nat→Nat→Nat→Native.Word)
    (result : KernelOutput.Result M)
    (claim : KernelPublic.Claim api c out tensor callerDevice cfg m target result)
    (b r h d : Nat) (byte : Fin 4) (hb : b<cfg.shape.batches) (hr : r<cfg.length b)
    (hh : h<cfg.shape.heads) (hd : d<cfg.shape.width) :
    api.observe result.memory result.tensor.device.index
      (result.tensor.dataPtr.toNat+4*cfg.shape.address (cfg.start b+r) h d+byte.val)=
        Physical.encode (target b r h d) byte :=
  KernelPublic.native_bytes api codec c out tensor callerDevice cfg m target result claim
    b r h d byte hb hr hh hd

theorem rotary_transform_signed_metadata {α : Type} [Inhabited α]
    (c : Checked.Call) (bytes : Physical.Bytes) (payload : Memory α) (other : State α)
    (accepted : check c bytes=Bool.true) :
    ProgramModel.MetadataBinding (Checked.configuration c bytes) (KernelEntry.lift c bytes payload other) :=
  KernelEntry.metadata c bytes payload other accepted

theorem rotary_transform_source_tiles (c : Checked.Call) (bytes : Physical.Bytes)
    (accepted : check c bytes=Bool.true) (launch : Host.empty c.host.shape=Bool.false) :
    ∀ shape∈SourceTiles.statementListShapes
      (TypedExecution.kernel c.host.shape c.host.abi.cumulative64 c.host.abi.offsets64).surfaceBody,
      shape.prod≤1048576 := KernelIndex.checked_tiles c bytes accepted launch

theorem rotary_transform_checked_indices (c : Checked.Call) (bytes : Physical.Bytes)
    (accepted : check c bytes=Bool.true) :
    IndexABI (Checked.configuration c bytes) := KernelIndex.checked_indices c bytes accepted

theorem rotary_transform_data_pointer (c : Checked.Call) (out : Binding.Output c)
    (bytes : Physical.Bytes) (checked : Binding.pointerCheck c out bytes=Bool.true)
    (launch : Host.empty c.host.shape=Bool.false) (offset : Nat)
    (inside : offset<Binding.dataBound c) (output : Bool) :
    let base := if output then out.tensor.base else c.input.base
    (BitVec.ofNat 64 base+BitVec.ofNat 64 offset*BitVec.ofNat 64 4).toNat=base+4*offset :=
  KernelIndex.data_pointer c out bytes checked launch offset inside output

theorem rotary_transform_coefficient_pointer (c : Checked.Call) (out : Binding.Output c)
    (bytes : Physical.Bytes) (checked : Binding.pointerCheck c out bytes=Bool.true)
    (launch : Host.empty c.host.shape=Bool.false) (batch : Fin c.host.shape.batches)
    (sine : Bool) (offset : Nat) (inside : offset<Binding.coefficientBound c bytes batch.val) :
    (BitVec.ofNat 64 (out.coefficientBase sine)+BitVec.ofNat 64 offset*BitVec.ofNat 64 4).toNat=
      out.coefficientBase sine+4*offset :=
  KernelIndex.coefficient_pointer c out bytes checked launch batch sine offset inside

theorem rotary_transform_correct {M β : Type} (api : API M ℝ β) (c : Checked.Call) (out : Binding.Output c)
    (inputs : RotaryTransform.Region→VeriTile.Torch.TensorMetadata) (binding : Inputs c out inputs)
    (tensor : VeriTile.Torch.TensorMetadata) (observed : Observes out.tensor tensor)
    (callerDevice : Nat) (bytes : Physical.Bytes) (m : M) (other : State ℝ)
    (accepted : check c bytes=Bool.true) (pointers : Binding.pointerCheck c out bytes=Bool.true) :
    let cfg := Checked.configuration c bytes
    ∃ final result,
      TypedGrid.run realOps cfg c.host.abi.cumulative64 c.host.abi.offsets64
        (realEntry api c out bytes m other)=some final ∧
      realInvoke api c out tensor callerDevice bytes m other=some result ∧
      Claim api c out tensor callerDevice cfg m (realFormula cfg (logicalInputs api c inputs m)) result := by
  let cfg := Checked.configuration c bytes
  let payload := RealWrapper.prepared c (inputMemory api c out m)
  have valid : cfg.Valid := Checked.config_valid c bytes (admitted c bytes accepted)
  obtain ⟨final,success,values,_⟩ := KernelEndpoints.real cfg valid (realEntry api c out bytes m other)
    (KernelEntry.metadata c bytes payload other accepted) (KernelEntry.suffix c bytes payload other)
    c.host.abi.cumulative64 c.host.abi.offsets64
  refine ⟨final,⟨tensor,out.tensor.object,callerDevice,serialize api c out id final m⟩,success,
    by
      simp only [realInvoke,accepted,pointers,Bool.and_self,↓reduceIte]
      exact congrArg (Option.map (fun state =>
        (⟨tensor,out.tensor.object,callerDevice,serialize api c out id state m⟩ : Result M))) success,?_⟩
  refine ⟨rfl,rfl,observed.shape.trans out.shape,?_,observed.device.trans out.device,rfl,
    fun hi => congrArg (fun t : Checked.Tensor => t.object) (out.inplace hi),?_,?_⟩
  · simpa [out.dtype] using observed.dtype
  · intro b r h d hb hr hh hd
    rw [tensor_key _ _ observed]
    have bound := Host.address_bound c.host.shape _ h d (row_bound cfg valid b r hb hr) hh hd
    change api.read (serialize api c out id final m)
      (Binding.key out.tensor (c.host.shape.address (cfg.start b+r) h d))=
        realFormula cfg (logicalInputs api c inputs m) b r h d
    rw [read_serialize api c out id final m _ bound]
    have value := values b r h d hb hr hh hd
    rw [show KernelAlgebra.memory (realEntry api c out bytes m other)=
        KernelAlgebra.memory (KernelEntry.lift c bytes payload other) from rfl,
      KernelEntry.real_formula,real_input_formula api c out inputs binding m cfg] at value
    exact value
  · exact serialize_frame api c out id final m

theorem rotary_transform_fp32 {M : Type} (api : Binding.API M Native.Word Physical.Byte) (codec : Binding.Codec api)
    (c : Checked.Call) (out : Binding.Output c)
    (inputs : RotaryTransform.Region→VeriTile.Torch.TensorMetadata) (binding : Inputs c out inputs)
    (tensor : VeriTile.Torch.TensorMetadata) (observed : Observes out.tensor tensor)
    (callerDevice : Nat) (p : Native.Instructions) (m : M) (other : State Native.Node)
    (accepted : check c (Binding.mirror api c out m)=Bool.true)
    (pointers : Binding.pointerCheck c out (Binding.mirror api c out m)=Bool.true) :
    let bytes := Binding.mirror api c out m
    let cfg := Checked.configuration c bytes
    ∃ final result,
      TypedGrid.run (Native.ops p) cfg c.host.abi.cumulative64 c.host.abi.offsets64
        (nativeEntry api c out m other)=some final ∧
      nativeInvoke api c out tensor callerDevice p m other=some result ∧
      Claim api c out tensor callerDevice cfg m (nativeFormula cfg p (logicalInputs api c inputs m)) result := by
  let bytes := Binding.mirror api c out m
  let cfg := Checked.configuration c bytes
  let payload := Physical.callerWords c.host.shape c.views bytes
  have valid : cfg.Valid := Checked.config_valid c bytes (admitted c bytes accepted)
  obtain ⟨final,success,values,_⟩ := KernelEndpoints.native cfg valid p (nativeEntry api c out m other)
    (KernelEntry.metadata c bytes (fun r i => Native.atom (payload r i)) other accepted)
    (KernelEntry.suffix c bytes (fun r i => Native.atom (payload r i)) other)
    c.host.abi.cumulative64 c.host.abi.offsets64
  refine ⟨final,⟨tensor,out.tensor.object,callerDevice,serialize api c out Native.Node.value final m⟩,success,
    by
      simp only [nativeInvoke,accepted,pointers,Bool.and_self,↓reduceIte]
      exact congrArg (Option.map (fun state =>
        (⟨tensor,out.tensor.object,callerDevice,serialize api c out Native.Node.value state m⟩ : Result M))) success,?_⟩
  refine ⟨rfl,rfl,observed.shape.trans out.shape,?_,observed.device.trans out.device,rfl,
    fun hi => congrArg (fun t : Checked.Tensor => t.object) (out.inplace hi),?_,?_⟩
  · simpa [out.dtype] using observed.dtype
  · intro b r h d hb hr hh hd
    rw [tensor_key _ _ observed]
    have bound := Host.address_bound c.host.shape _ h d (row_bound cfg valid b r hb hr) hh hd
    change api.read (serialize api c out Native.Node.value final m)
      (Binding.key out.tensor (c.host.shape.address (cfg.start b+r) h d))=
        nativeFormula cfg p (logicalInputs api c inputs m) b r h d
    rw [read_serialize api c out Native.Node.value final m _ bound]
    have value := values b r h d hb hr hh hd
    rw [show KernelAlgebra.memory (nativeEntry api c out m other)=
        KernelAlgebra.memory (KernelEntry.lift c bytes (fun r i => Native.atom (payload r i)) other) from rfl,
      KernelEntry.native_formula,native_input_formula api codec c out inputs binding m cfg p] at value
    exact value
  · exact serialize_frame api c out Native.Node.value final m

#axiomsClean rotary_transform_correct
#axiomsClean rotary_transform_fp32
#axiomsClean rotary_transform_output_bytes
#axiomsClean rotary_transform_signed_metadata
#axiomsClean rotary_transform_source_tiles
#axiomsClean rotary_transform_checked_indices
#axiomsClean rotary_transform_data_pointer
#axiomsClean rotary_transform_coefficient_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.TensorBridge
