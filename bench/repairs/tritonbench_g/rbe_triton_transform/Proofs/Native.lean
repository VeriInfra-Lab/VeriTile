import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Memory

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000

structure Tensor where
  handle : Nat
  device : Nat
  base : Nat
  allocationEnd : Nat
  shape : List Nat
  strides : List Nat
  dtype : String
  elementBytes : Nat
  storageBase : Nat
  storageBytes : Nat
  storageOffset : Nat

def logicalAddress (t : Tensor) (b m c : Nat) :=
  t.base+4*(b*t.strides[0]!+m*t.strides[1]!+c*t.strides[2]!)
def logicalRead (api : Memory.API β Store Byte) (memory : Store) (t : Tensor) (b m c : Nat) :=
  api.read memory (logicalAddress t b m c)

/-- Same-call ordinary contiguous Tensor / empty_like contract, with only
local input reads and allocation frame. It assumes no kernel result. -/
structure Preparation {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β → α)
    (r : Request) (inputRegion : RegionName) (initial : State α) where
  before : Store
  memory : Store
  input : Tensor
  output : Tensor
  caller : Nat → Prop
  inputShape : input.shape = [r.batches,r.rows,r.columns]
  outputShape : output.shape = [r.batches,r.rows,r.columns]
  inputType : input.dtype = "float32" ∧ input.elementBytes=4
  outputType : output.dtype = "float32" ∧ output.elementBytes=4
  sameDevice : output.device=input.device
  inputAddress : ∀ b m c, b < r.batches → m < r.rows → c < r.columns →
    logicalAddress input b m c = input.base+4*r.offset b m c
  outputAddress : ∀ b m c, b < r.batches → m < r.rows → c < r.columns →
    logicalAddress output b m c = output.base+4*r.offset b m c
  inputBinding : ∀ b m c, b < r.batches → m < r.rows → c < r.columns →
    (initial.mem inputRegion (r.offset b m c)).read .fp32 =
      embed (api.read before (input.base+4*r.offset b m c))
  preservedInput : ∀ o, o < r.size → api.read memory (input.base+4*o)=api.read before (input.base+4*o)
  inputFits : input.base+4*r.size ≤ input.allocationEnd ∧ input.allocationEnd ≤ 2^64
  outputFits : output.base+4*r.size ≤ output.allocationEnd ∧ output.allocationEnd ≤ 2^64
  separate : output.base+4*r.size ≤ input.base ∨ input.base+4*r.size ≤ output.base
  frame : ∀ b, caller b → api.byte memory b=api.byte before b
  fresh : ∀ b, caller b → b < output.base ∨ output.base+4*r.size ≤ b
  inputBacking : input.allocationEnd=input.storageBase+input.storageBytes
  inputPointer : 0<r.size → input.base=input.storageBase+4*input.storageOffset
  inputCaller : ∀ b, input.storageBase ≤ b → b < input.storageBase+input.storageBytes → caller b
  outputAligned : output.base%4=0

structure Returned (Store : Type) where
  memory : Store
  tensor : Tensor

def entry {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β → α) (project : α → β)
    (ops : Operations α) (r : Request) (input output : RegionName) (s : State α)
    (call : Preparation api embed r input s) : Returned Store :=
  ⟨Memory.run api project ops r input output s call.output.base call.memory,call.output⟩

/-- Widened source arithmetic. Active addresses are exact; inactive pointer
expressions may wrap because their masks prohibit memory events. -/
def IndexABI {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β → α)
    (r : Request) (input : RegionName) (s : State α) (call : Preparation api embed r input s) : Prop :=
  (0<r.size → r.batches < 2^63 ∧ r.rows+1 < 2^63 ∧ r.columns+1023 < 2^63 ∧
    r.rows*r.columns < 2^63 ∧ grid r < 2^63) ∧
  ∀ l ∈ schedule r,
    row r l<2^63 ∧ evenColumn r l+1<2^63 ∧
    -(2^63 : Int) ≤ r.position+(row r l : Int) ∧ r.position+(row r l : Int) < 2^63 ∧
    (active r l → r.offset (batch l) (row r l) (column r l) < 2^63 ∧
      call.input.base+4*r.offset (batch l) (row r l) (column r l)+4 ≤ 2^64 ∧
      call.output.base+4*r.offset (batch l) (row r l) (column r l)+4 ≤ 2^64)

theorem prefix_inputs {α : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β → α) (project : α → β) (ops : Operations α) (r : Request)
    (input output : RegionName) (s : State α) (call : Preparation api embed r input s)
    (xs : List Location) (o : Nat) (ho : o < r.size) :
    api.read (Memory.runWrites api (Memory.events project ops r input output s call.output.base xs) call.memory)
      (call.input.base+4*o) = api.read call.before (call.input.base+4*o) := by
  rw [Memory.writes_read_frame]
  · exact call.preservedInput o ho
  · intro e he
    obtain ⟨l,hl,ha,rfl⟩ := (Memory.mem_events ..).mp he
    have ht := offset_lt r _ _ _ ha.1 ha.2.1 ha.2.2
    have hs := call.separate
    dsimp only
    omega

theorem source_load_binding {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β → α) (r : Request) (input : RegionName) (s : State α)
    (call : Preparation api embed r input s) (b m c : Nat)
    (hb : b < r.batches) (hm : m < r.rows) (hc : c < r.columns) :
    (s.mem input (r.offset b m c)).read .fp32 = embed (logicalRead api call.memory call.input b m c) := by
  rw [logicalRead,call.inputAddress b m c hb hm hc,
    call.preservedInput _ (offset_lt r b m c hb hm hc)]
  exact call.inputBinding b m c hb hm hc

theorem whole {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β → α) (project : α → β)
    (ops : Operations α) (r : Request) (input output : RegionName) (s : State α)
    (call : Preparation api embed r input s) :
    (entry api embed project ops r input output s call).tensor=call.output ∧
    (entry api embed project ops r input output s call).tensor.shape=[r.batches,r.rows,r.columns] ∧
    (∀ b m c, b < r.batches → m < r.rows → c < r.columns →
      logicalRead api (entry api embed project ops r input output s call).memory
        (entry api embed project ops r input output s call).tensor b m c =
          project (scalarResult ops r (fun o => (s.mem input o).read .fp32) b m c)) ∧
    (∀ b m c, b < r.batches → m < r.rows → c < r.columns → ∀ k : Fin 4,
      api.byte (entry api embed project ops r input output s call).memory
        (logicalAddress (entry api embed project ops r input output s call).tensor b m c+k.val) =
          api.encode (project (scalarResult ops r (fun o => (s.mem input o).read .fp32) b m c)) k) ∧
    (∀ b, call.caller b → api.byte (entry api embed project ops r input output s call).memory b=api.byte call.before b) := by
  refine ⟨rfl,call.outputShape,?_,?_,?_⟩
  · intro b m c hb hm hc
    change api.read (Memory.run api project ops r input output s call.output.base call.memory) (logicalAddress call.output b m c) = _
    rw [call.outputAddress b m c hb hm hc]
    exact Memory.readback api project ops r input output s call.output.base call.memory b m c hb hm hc
  · intro b m c hb hm hc k
    rw [api.read_bytes]
    change api.encode (api.read (Memory.run api project ops r input output s call.output.base call.memory) (logicalAddress call.output b m c)) k = _
    rw [call.outputAddress b m c hb hm hc,Memory.readback api project ops r input output s call.output.base call.memory b m c hb hm hc]
  · intro b hb
    change api.byte (Memory.run api project ops r input output s call.output.base call.memory) b = _
    rw [Memory.byte_frame api project ops r input output s call.output.base call.memory b (call.fresh b hb)]
    exact call.frame b hb

theorem schedule_empty (r : Request) (h : r.size=0) : schedule r=[] := by
  simp only [Request.size,Nat.mul_eq_zero] at h
  rcases h with (hb | hm) | hc
  · simp [schedule,hb]
  · simp [schedule,grid,rowParts,hm]
  · simp [schedule,grid,parts,hc]

theorem empty_return {α : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β → α) (project : α → β) (ops : Operations α) (r : Request)
    (input output : RegionName) (s : State α) (call : Preparation api embed r input s) (h : r.size=0) :
    entry api embed project ops r input output s call=⟨call.memory,call.output⟩ := by
  simp [entry,Memory.run,schedule_empty r h,Memory.events,Memory.runWrites]

#specNonCircular Preparation avoiding [entry,Memory.run,FP.Structural.exec]
#axiomsClean prefix_inputs
#axiomsClean source_load_binding
#axiomsClean whole
#axiomsClean empty_return
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Native
