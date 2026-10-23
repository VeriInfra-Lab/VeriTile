import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Checked
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Trace
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.Binding
noncomputable section
open Checked

/-- Four physical little-endian bytes at the actual Tensor data pointer. -/
def tensorWord (x : Tensor) (memory : Bytes) (b : Nat) : BitVec 32 :=
  wordOf (fun j => memory (x.metadata.dataPtr.toNat+4*b+j.val))

def run (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) : Bytes :=
  executeBytes (byteTrace c layout
    (Native.nativeForward a compiler c plan scale (loadWords c layout memory))) memory

def invoke (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (x : Inputs) (plan : Plans c) (scale : BitVec 32) (pythonScale : PythonNumber)
    (caller : VeriTile.Torch.Device) (layout : Layout c) (alloc : Allocations c x layout.base)
    (memory : Bytes) : Option (Returned Bytes) :=
  if admit c x then some (returned c x layout.base alloc pythonScale caller
    (run a compiler c plan scale layout memory)) else none

def Outside (c : Shape) (layout : Layout c) (addr : Nat) : Prop :=
  ∀ r b,WholeOutput c r b → ∀ j : Fin 4,addr≠layout.base r+4*b+j.val

def targetOutput (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) (z t f : Nat) :=
  Native.observe a compiler (scalarOutput Native.sourceOps c plan
    (inputs c (Native.entry (loadWords c layout memory))) (.word .outside 0 scale) z t f)

def targetFinal (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (layout : Layout c) (memory : Bytes) (z e f : Nat) :=
  Native.observe a compiler (.store32 (scalarState Native.sourceOps c
    (inputs c (Native.entry (loadWords c layout memory))) z e f c.T))

structure NativeResult (actual target : Native.Instructions) (compiler : Native.Compiler)
    (c : Shape) (x : Inputs) (plan : Plans c) (scale : BitVec 32) (pythonScale : PythonNumber)
    (caller : VeriTile.Torch.Device) (layout : Layout c) (alloc : Allocations c x layout.base)
    (memory : Bytes) (result : Returned Bytes) : Prop where
  output : result.output=alloc.output
  metadata : result.output.Valid ∧ result.output.metadata.shape=tensorShape c .result ∧
    result.output.metadata.device=(x.tensor .q).metadata.device ∧
    result.output.metadata.dataPtr.toNat=layout.base .result
  terminal : result.finalState=(if c.final then some alloc.terminal else none)
  terminalMetadata : c.final=true → alloc.terminal.Valid ∧ alloc.terminal.metadata.shape=tensorShape c .ht ∧
    alloc.terminal.metadata.device=(x.tensor .q).metadata.device ∧
    alloc.terminal.metadata.dataPtr.toNat=layout.base .ht
  context : result.saved=saved x ∧ result.savedScale=pythonScale ∧
    result.savedReverse=c.reverse ∧ result.callerDevice=caller
  execution : result.memory=run actual compiler c plan scale layout memory
  originalLoads : ∀ r∈inputRegions c,∀ b,
    loadWords c layout memory r b=tensorWord (x.tensor r) memory b
  outputWords : ∀ z<c.Z,∀ t<c.T,∀ f<c.V,
    tensorWord result.output result.memory (resultAddress c z t f)=
      targetOutput target compiler c plan scale layout memory z t f
  outputBytes : ∀ z<c.Z,∀ t<c.T,∀ f<c.V,∀ j : Fin 4,
    result.memory (result.output.metadata.dataPtr.toNat+4*resultAddress c z t f+j.val)=
      byteOf (targetOutput target compiler c plan scale layout memory z t f) j
  finalWords : c.final=true → ∀ z<c.Z,∀ e<c.K,∀ f<c.V,
    tensorWord alloc.terminal result.memory (stateAddress c z e f)=
      targetFinal target compiler c layout memory z e f
  finalBytes : c.final=true → ∀ z<c.Z,∀ e<c.K,∀ f<c.V,∀ j : Fin 4,
    result.memory (alloc.terminal.metadata.dataPtr.toNat+4*stateAddress c z e f+j.val)=
      byteOf (targetFinal target compiler c layout memory z e f) j
  inputFrame : ∀ r,InputRegion r → ∀ b,b<extent c r → ∀ j : Fin 4,
    result.memory (layout.base r+4*b+j.val)=memory (layout.base r+4*b+j.val)
  exterior : ∀ addr,Outside c layout addr → result.memory addr=memory addr
  formedInputs : ∀ r∈inputRegions c,0<(x.tensor r).metadata.numel →
    ∀ offset : Int,-(negative c r:Int)≤offset → offset ≤ indexLimit c →
    ∀ j : Fin 4,AddressFits (x.tensor r) offset j
  formedScratch : 0<alloc.scratch.metadata.numel →
    ∀ offset : Int,-(negative c .o:Int)≤offset → offset ≤ indexLimit c →
    ∀ j : Fin 4,AddressFits alloc.scratch offset j
  formedTerminal : c.final=true → 0<alloc.terminal.metadata.numel →
    ∀ offset : Int,-(negative c .ht:Int)≤offset → offset ≤ indexLimit c →
    ∀ j : Fin 4,AddressFits alloc.terminal offset j

theorem complete (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (x : Inputs) (plan : Plans c) (scale : BitVec 32) (pythonScale : PythonNumber)
    (caller : VeriTile.Torch.Device) (layout : Layout c) (alloc : Allocations c x layout.base)
    (direct : DirectInputs c x layout.base) (memory : Bytes) (admitted : admit c x=true) :
    ∃ result,invoke a compiler c x plan scale pythonScale caller layout alloc memory=some result ∧
      NativeResult a a compiler c x plan scale pythonScale caller layout alloc memory result := by
  have hc := domain c x admitted
  have run_eq : run a compiler c plan scale layout memory=
      physicalForward a compiler c plan scale layout memory := native_trace a compiler c hc.sizes plan scale layout memory
  refine ⟨returned c x layout.base alloc pythonScale caller (run a compiler c plan scale layout memory),
    by simp only [invoke,admitted,if_true],?_⟩
  constructor
  · rfl
  · exact alloc.outputMetadata
  · rfl
  · exact alloc.terminalMetadata
  · exact ⟨rfl,rfl,rfl,rfl⟩
  · rfl
  · intro r hr b
    simp only [loadWords,tensorWord,direct.pointers r hr]
  · intro z hz t ht f hf
    change tensorWord alloc.output (run a compiler c plan scale layout memory) _ = _
    simp only [tensorWord,alloc.outputMetadata.2.2.2,run_eq]
    exact physical_returned a compiler c hc.sizes plan scale layout memory z t f hz ht hf
  · intro z hz t ht f hf j
    change run a compiler c plan scale layout memory (alloc.output.metadata.dataPtr.toNat+4*resultAddress c z t f+j.val) = _
    rw [alloc.outputMetadata.2.2.2,run_eq]
    unfold physicalForward
    rw [scatter_byte c layout memory _ .result _ j (Or.inr ⟨rfl,result_bound c z t f hz ht hf⟩)]
    rw [Native.returned a compiler c hc.sizes plan scale _ z t f hz ht hf]
    rfl
  · intro hs z hz e he f hf
    change tensorWord alloc.terminal (run a compiler c plan scale layout memory) _ = _
    simp only [tensorWord,(alloc.terminalMetadata hs).2.2.2,run_eq]
    exact physical_terminal a compiler c hc.sizes plan scale layout memory hs z e f hz he hf
  · intro hs z hz e he f hf j
    change run a compiler c plan scale layout memory (alloc.terminal.metadata.dataPtr.toNat+4*stateAddress c z e f+j.val) = _
    rw [(alloc.terminalMetadata hs).2.2.2,run_eq]
    unfold physicalForward
    rw [scatter_byte c layout memory _ .ht _ j
      (Or.inl (Or.inr ⟨hs,rfl,state_address_bound c z e f hz he hf⟩))]
    rw [Native.terminal a compiler c hc.sizes plan scale _ hs z e f hz he hf]
    rfl
  · intro r hr b hb j
    change run a compiler c plan scale layout memory _ = _
    rw [run_eq]
    exact physical_inputs a compiler c plan scale layout memory r hr b hb j
  · intro addr outside
    change run a compiler c plan scale layout memory addr = _
    rw [run_eq]
    apply physical_frame
    intro r b j decoded hw
    exact outside r b hw j (layout.sound addr r b j decoded).2
  · intro r hr active offset low high j
    exact formed_address c r (x.tensor r)
      (((admit_iff c x).mp admitted).2.2.2.2 r hr).2.2.2 active offset low high j
  · intro active offset low high j
    exact formed_address c .o alloc.scratch alloc.scratchChecked active offset low high j
  · intro hs active offset low high j
    exact formed_address c .ht alloc.terminal (alloc.terminalChecked hs) active offset low high j

theorem complete_if_local (actual target : Native.Instructions)
    (localAgreement : Native.InstructionAgreement actual target) (compiler : Native.Compiler) (c : Shape)
    (x : Inputs) (plan : Plans c) (scale : BitVec 32) (pythonScale : PythonNumber)
    (caller : VeriTile.Torch.Device) (layout : Layout c) (alloc : Allocations c x layout.base)
    (direct : DirectInputs c x layout.base) (memory : Bytes) (admitted : admit c x=true) :
    ∃ result,invoke actual compiler c x plan scale pythonScale caller layout alloc memory=some result ∧
      NativeResult actual target compiler c x plan scale pythonScale caller layout alloc memory result := by
  obtain ⟨result,inv,h⟩ := complete actual compiler c x plan scale pythonScale caller layout alloc direct memory admitted
  have same : ∀ term,Native.observe actual compiler term=Native.observe target compiler term :=
    fun term => Native.eval_agreement actual target localAgreement _
  refine ⟨result,inv,{h with outputWords:=?_,outputBytes:=?_,finalWords:=?_,finalBytes:=?_}⟩
  · intro z hz t ht f hf
    rw [h.outputWords z hz t ht f hf];exact same _
  · intro z hz t ht f hf j
    rw [h.outputBytes z hz t ht f hf j];unfold targetOutput;rw [same]
  · intro hs z hz e he f hf
    rw [h.finalWords hs z hz e he f hf];exact same _
  · intro hs z hz e he f hf j
    rw [h.finalBytes hs z hz e he f hf j];unfold targetFinal;rw [same]

#axiomsClean complete
#axiomsClean complete_if_local
#specNonCircular targetOutput avoiding [invoke,run,runLoop,runProgram,runGrid,forward,Native.nativeForward]
#specNonCircular targetFinal avoiding [invoke,run,runLoop,runProgram,runGrid,forward,Native.nativeForward]
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.Binding
