import bench.tritonbench_g.rotary_transform.Proofs.KernelOutput
import bench.tritonbench_g.rotary_transform.Proofs.SourceTiles

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelPublic
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Triton FP.Structural
open KernelOutput
noncomputable section

abbrev API := VeriTile.Bench.Repairs.SwigluTriton.Memory.Ops

def tensorKey (t : VeriTile.Torch.TensorMetadata) (i : Nat) : Key :=
  ⟨t.device.index,t.dataPtr.toNat+4*i⟩

theorem tensor_key (t : Checked.Tensor) (handle : VeriTile.Torch.TensorMetadata)
    (observed : Observes t handle) (i : Nat) : tensorKey handle i=Binding.key t i := by
  simp [tensorKey,Binding.key,observed.pointer,observed.device]

/-- Independent input-only view through original caller Tensor handles.
Coefficient coordinates use their original positive strides. -/
def logicalInputs {M α β : Type} (api : API M α β) (c : Checked.Call)
    (tensor : RotaryTransform.Region→VeriTile.Torch.TensorMetadata) (m : M) : Memory α :=
  fun r i => if r=.cosine then api.read m (tensorKey (tensor .cosine) (c.views.cosineAt c.host.shape i))
    else if r=.sine then api.read m (tensorKey (tensor .sine) (c.views.sineAt c.host.shape i))
    else api.read m (tensorKey (tensor .input) i)

theorem real_input_formula {M β : Type} (api : API M ℝ β) (c : Checked.Call)
    (out : Binding.Output c) (tensor : RotaryTransform.Region→VeriTile.Torch.TensorMetadata)
    (binding : Inputs c out tensor) (m : M) (cfg : Config) :
    realFormula cfg (RealWrapper.prepared c (inputMemory api c out m))=
      realFormula cfg (logicalInputs api c tensor m) := by
  have hx := binding.observed .input (by simp [handles])
  have hc := binding.observed .cosine (by simp [handles])
  have hs := binding.observed .sine (by simp [handles])
  funext b r h d
  simp [realFormula,RealWrapper.prepared,inputMemory,logicalInputs,tensorKey,Binding.key,
    descriptor] at hx hc hs ⊢
  rw [hx.pointer,hx.device,hc.pointer,hc.device,hs.pointer,hs.device]

theorem native_input_formula {M : Type} (api : Binding.API M Native.Word Physical.Byte)
    (codec : Binding.Codec api) (c : Checked.Call) (out : Binding.Output c)
    (tensor : RotaryTransform.Region→VeriTile.Torch.TensorMetadata) (binding : Inputs c out tensor)
    (m : M) (cfg : Config) (p : Native.Instructions) :
    nativeFormula cfg p (Physical.callerWords c.host.shape c.views (Binding.mirror api c out m))=
      nativeFormula cfg p (logicalInputs api c tensor m) := by
  have hx := binding.observed .input (by simp [handles])
  have hc := binding.observed .cosine (by simp [handles])
  have hs := binding.observed .sine (by simp [handles])
  funext b r h d
  simp only [nativeFormula,Physical.callerWords,reduceCtorEq,if_false,if_true,
    native_input_read api codec c out m]
  simp [inputMemory,logicalInputs,tensorKey,Binding.key,descriptor] at hx hc hs ⊢
  rw [hx.pointer,hx.device,hc.pointer,hc.device,hs.pointer,hs.device]

structure Claim {M α β : Type} (api : API M α β) (c : Checked.Call) (out : Binding.Output c)
    (tensor : VeriTile.Torch.TensorMetadata) (callerDevice : Nat) (cfg : Config) (m : M)
    (target : Nat→Nat→Nat→Nat→α) (result : Result M) : Prop where
  returned : result.tensor=tensor
  object : result.object=out.tensor.object
  shape : result.tensor.shape=c.input.shape
  dtype : result.tensor.dtype=.floating .fp32
  device : result.tensor.device.index=c.host.device
  restored : result.deviceRestored=callerDevice
  inplace : c.host.shape.inplace=Bool.true → result.object=c.input.object
  values : ∀ b r h d,b<cfg.shape.batches → r<cfg.length b → h<cfg.shape.heads → d<cfg.shape.width →
    api.read result.memory (tensorKey result.tensor (cfg.shape.address (cfg.start b+r) h d))=target b r h d
  frame : ∀ owner address,Binding.Outside c out owner address →
    api.observe result.memory owner address=api.observe m owner address

/-- Complete arbitrary-Real rotation from original Tensor reads, through the
full shared quotation, into returned Tensor storage and its exterior frame.
The existing allocation/identity and local memory API laws are the boundary. -/
theorem real_complete {M β : Type} (api : API M ℝ β) (c : Checked.Call) (out : Binding.Output c)
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

/-- Exact native raw-word graph through the same full quotation. The four
primitive callbacks and the stated compiler fusion pattern are explicit
model boundaries; every raw FP32 input pattern is retained. -/
theorem native_complete {M : Type} (api : Binding.API M Native.Word Physical.Byte) (codec : Binding.Codec api)
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

/-- Local codec turns the proven returned word into all four output bytes. -/
theorem native_bytes {M : Type} (api : Binding.API M Native.Word Physical.Byte) (codec : Binding.Codec api)
    (c : Checked.Call) (out : Binding.Output c) (tensor : VeriTile.Torch.TensorMetadata)
    (callerDevice : Nat) (cfg : Config) (m : M) (target : Nat→Nat→Nat→Nat→Native.Word)
    (result : Result M) (claim : Claim api c out tensor callerDevice cfg m target result)
    (b r h d : Nat) (byte : Fin 4) (hb : b<cfg.shape.batches) (hr : r<cfg.length b)
    (hh : h<cfg.shape.heads) (hd : d<cfg.shape.width) :
    api.observe result.memory result.tensor.device.index
      (result.tensor.dataPtr.toNat+4*cfg.shape.address (cfg.start b+r) h d+byte.val)=
        Physical.encode (target b r h d) byte := by
  change api.observe result.memory (tensorKey result.tensor (cfg.shape.address (cfg.start b+r) h d)).owner
    ((tensorKey result.tensor (cfg.shape.address (cfg.start b+r) h d)).address+byte.val)=_
  rw [codec.bytes,claim.values b r h d hb hr hh hd]

#axiomsClean real_complete
#axiomsClean native_complete
#axiomsClean native_bytes
#specNonCircular logicalInputs avoiding [TypedGrid.run,ProgramModel.result,realInvoke,nativeInvoke]
#specNonCircular realFormula avoiding [TypedGrid.run,ProgramModel.result,source,sourcePair,pairValues]
#specNonCircular nativeFormula avoiding [TypedGrid.run,ProgramModel.result,Native.graph,Native.pairWord,Native.ops]
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelPublic
