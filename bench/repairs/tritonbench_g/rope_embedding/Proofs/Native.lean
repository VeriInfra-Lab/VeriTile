import bench.repairs.tritonbench_g.rope_embedding.Proofs.Memory
namespace VeriTile.Bench.Repairs.RopeEmbedding.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

structure Tensor where
  handle : Nat
  storage : Nat
  allocationBase : Nat
  device : Nat
  base : Nat
  allocationEnd : Nat
  shape : List Nat
  strides : List Nat
  dtype : String
  elementBytes : Nat
  (strided cuda contiguous materialized : Bool)

def logicalAddress (t : Tensor) (b h m c : Nat) :=
  t.base+4*(b*t.strides[0]!+h*t.strides[1]!+m*t.strides[2]!+c*t.strides[3]!)
def logicalRead (api : Memory.API β Store Byte) (memory : Store) (t : Tensor) (b h m c : Nat) :=
  api.read memory (logicalAddress t b h m c)

def tableShape (half rows : Nat) (shape : List Nat) : Prop :=
  shape=[rows,half] ∨ (half=1 ∧ shape=[rows])
instance (half rows : Nat) (shape : List Nat) : Decidable (tableShape half rows shape) :=
  inferInstanceAs (Decidable (_∨_))

def canReuse (r : Request) : Prop := r.size=0 ∨ r.sequence≤1 ∨ r.heads≤1
instance (r : Request) : Decidable (canReuse r) := inferInstanceAs (Decidable (_∨_∨_))

/-- Ordinary original transpose/clone/reshape preparation and returned view.
`cloneInput` selects the original forward-only clone. A reshape-only call may
reuse its input storage. Coefficients are actual squeeze views in forward and
the supplied objects in backward, with no added copy or cached snapshot. -/
structure Preparation {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (r : Request) (initial : State α) where
  before : Store
  memory : Store
  (input output cosine sine cosineInput sineInput : Tensor)
  (cloneInput squeezed reused : Bool)
  (cosineRows sineRows : Nat)
  caller : Nat→Prop
  callerBinding : ∀ b,caller b ↔
    (input.allocationBase≤b ∧ b < input.allocationEnd) ∨
    (cosineInput.allocationBase≤b ∧ b < cosineInput.allocationEnd) ∨
    (sineInput.allocationBase≤b ∧ b < sineInput.allocationEnd)
  coefficientViews :
    (if squeezed then cosine.shape=cosineInput.shape.filter (· != 1) ∧
      sine.shape=sineInput.shape.filter (· != 1) ∧
      cosine.storage=cosineInput.storage ∧ sine.storage=sineInput.storage ∧
      cosine.base=cosineInput.base ∧ sine.base=sineInput.base
    else cosine=cosineInput ∧ sine=sineInput)
  reuseCondition : reused=(!cloneInput && decide (canReuse r))
  reuseIdentity : reused=Bool.true → output.storage=input.storage ∧ output.base=input.base
  inputShape : input.shape=[r.batches,r.heads,r.sequence,r.dimension]
  outputShape : output.shape=[r.batches,r.heads,r.sequence,r.dimension]
  cosineShape : tableShape r.half cosineRows cosine.shape
  sineShape : tableShape r.half sineRows sine.shape
  tableLengths : r.sequence≤cosineRows ∧ r.sequence≤sineRows
  inputType : input.dtype="float32" ∧ input.elementBytes=4
  outputType : output.dtype="float32" ∧ output.elementBytes=4
  cosineType : cosine.dtype="float32" ∧ cosine.elementBytes=4
  sineType : sine.dtype="float32" ∧ sine.elementBytes=4
  devices : output.device=input.device ∧ cosine.device=input.device ∧ sine.device=input.device
  cosineAddress : ∀ a j,a<cosineRows → j<r.half →
    a*cosine.strides[0]!+j*cosine.strides[1]! =a*r.half+j
  sineAddress : ∀ a j,a<sineRows → j<r.half →
    a*sine.strides[0]!+j*sine.strides[1]! =a*r.half+j
  inputAddress : ∀ b h m c,b<r.batches → h<r.heads → m<r.sequence → c<r.dimension →
    logicalAddress input b h m c=input.base+4*r.external b h m c
  outputAddress : ∀ b h m c,b<r.batches → h<r.heads → m<r.sequence → c<r.dimension →
    logicalAddress output b h m c=output.base+4*r.offset (b*r.sequence+m) h c
  inputBinding : ∀ a h c,a<r.rows → h<r.heads → c<r.dimension →
    (initial.mem "Q" (r.offset a h c)).read .real=
      embed (api.read before (input.base+4*r.external (a/r.sequence) h (a%r.sequence) c))
  preparedInput : ∀ a h c,a<r.rows → h<r.heads → c<r.dimension →
    api.read memory (output.base+4*r.offset a h c)=
      api.read before (input.base+4*r.external (a/r.sequence) h (a%r.sequence) c)
  cosineBinding : ∀ k,k<r.sequence*r.half → (initial.mem "C" k).read .real=embed (api.read before (cosine.base+4*k))
  sineBinding : ∀ k,k<r.sequence*r.half → (initial.mem "S" k).read .real=embed (api.read before (sine.base+4*k))
  preservedCosine : ∀ k,k<r.sequence*r.half → api.read memory (cosine.base+4*k)=api.read before (cosine.base+4*k)
  preservedSine : ∀ k,k<r.sequence*r.half → api.read memory (sine.base+4*k)=api.read before (sine.base+4*k)
  inputFits : input.base+4*r.size≤ input.allocationEnd ∧ input.allocationEnd≤2^64
  outputFits : output.base+4*r.size≤output.allocationEnd ∧ output.allocationEnd≤2^64
  cosineFits : cosine.base+4*cosineRows*r.half≤cosine.allocationEnd ∧ cosine.allocationEnd≤2^64
  sineFits : sine.base+4*sineRows*r.half≤sine.allocationEnd ∧ sine.allocationEnd≤2^64
  separateInput : reused=Bool.false → output.base+4*r.size≤ input.base ∨ input.base+4*r.size≤output.base
  separateCosine : output.base+4*r.size≤cosine.base ∨ cosine.base+4*(r.sequence*r.half)≤output.base
  separateSine : output.base+4*r.size≤sine.base ∨ sine.base+4*(r.sequence*r.half)≤output.base
  frame : ∀ b,caller b → api.byte memory b=api.byte before b
  fresh : reused=Bool.false → ∀ b,caller b → b<output.base ∨ output.base+4*r.size≤b
  inputOwned : ∀ b,input.base≤b → b< input.base+4*r.size → caller b
  cosineOwned : ∀ b,cosine.base≤b → b<cosine.base+4*(cosineRows*r.half) → caller b
  sineOwned : ∀ b,sine.base≤b → b<sine.base+4*(sineRows*r.half) → caller b

structure Returned (Store : Type) where
  memory : Store
  tensor : Tensor
structure ForwardReturned (Store : Type) extends Returned Store where
  cosine : Tensor
  sine : Tensor
  groups : Nat
  block : Nat
  warps : Nat

def entry {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (backward : Bool) (s : State α)
    (call : Preparation api embed r s) : Returned Store :=
  ⟨Memory.run api project op r backward s call.output.base call.memory,call.output⟩
def forward {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (s : State α) (call : Preparation api embed r s) : ForwardReturned Store :=
  ⟨entry api embed project op r Bool.false s call,call.cosine,call.sine,r.groups,r.block,r.warps⟩
def backward {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (s : State α) (call : Preparation api embed r s)
    (groups block warps : Nat) : Option (Returned Store) :=
  if (groups,block,warps)=(r.groups,r.block,r.warps) then some (entry api embed project op r Bool.true s call) else none

/-- Input-only range domain for the observed i32 specialization. All formed
pointers, including masked padded lanes and the second half, are represented. -/
def IndexABI {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (r : Request) (s : State α) (call : Preparation api embed r s) : Prop :=
  r.block≤65536 ∧ r.batches<2^31 ∧ r.heads+3<2^31 ∧ r.sequence<2^31 ∧ r.dimension<2^31 ∧
  r.rows<2^31 ∧ r.heads*r.dimension<2^31 ∧ r.groups<2^31 ∧
  ∀ l∈schedule r,group l*4+4<2^31 ∧
    r.offset (row l) (head l) ((lane l).1.val+r.half)<2^31 ∧
    row l%r.sequence*r.half+(lane l).1.val<2^31 ∧
    call.output.base+4*r.offset (row l) (head l) ((lane l).1.val+r.half)<2^64 ∧
    call.cosine.base+4*(row l%r.sequence*r.half+(lane l).1.val)<2^64 ∧
    call.sine.base+4*(row l%r.sequence*r.half+(lane l).1.val)<2^64

theorem row_bound (r : Request) (b m : Nat) (hb : b<r.batches) (hm : m<r.sequence) : b*r.sequence+m<r.rows := by
  have := Nat.mul_le_mul_right r.sequence (show b+1≤r.batches by omega)
  rw [Nat.add_mul,Nat.one_mul] at this
  unfold Request.rows
  omega

theorem whole {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (back : Bool) (s : State α) (call : Preparation api embed r s)
    (roundtrip : ∀ w,project (embed w)=w) :
    (entry api embed project op r back s call).tensor=call.output ∧
    (entry api embed project op r back s call).tensor.shape=[r.batches,r.heads,r.sequence,r.dimension] ∧
    (∀ b h m c,b<r.batches → h<r.heads → m<r.sequence → c<r.dimension →
      logicalRead api (entry api embed project op r back s call).memory
        (entry api embed project op r back s call).tensor b h m c=
          project (scalarResult op r back (fun o => (s.mem "Q" o).read .real)
            (fun o => (s.mem "C" o).read .real) (fun o => (s.mem "S" o).read .real) (b*r.sequence+m) h c)) ∧
    (∀ b h m c,b<r.batches → h<r.heads → m<r.sequence → c<r.dimension → ∀ k : Fin 4,
      api.byte (entry api embed project op r back s call).memory
        (logicalAddress (entry api embed project op r back s call).tensor b h m c+k.val)=
          api.encode (project (scalarResult op r back (fun o => (s.mem "Q" o).read .real)
            (fun o => (s.mem "C" o).read .real) (fun o => (s.mem "S" o).read .real) (b*r.sequence+m) h c)) k) ∧
    (∀ b,call.caller b → (b<call.output.base ∨ call.output.base+4*r.size≤b) →
      api.byte (entry api embed project op r back s call).memory b=api.byte call.before b) := by
  have value : ∀ b h m c,b<r.batches → h<r.heads → m<r.sequence → c<r.dimension →
      logicalRead api (entry api embed project op r back s call).memory call.output b h m c=
        project (scalarResult op r back (fun o => (s.mem "Q" o).read .real)
          (fun o => (s.mem "C" o).read .real) (fun o => (s.mem "S" o).read .real) (b*r.sequence+m) h c) := by
    intro b h m c hb hh hm hc
    change api.read (Memory.run api project op r back s call.output.base call.memory) (logicalAddress call.output b h m c)=_
    rw [call.outputAddress b h m c hb hh hm hc]
    have ha := row_bound r b m hb hm
    by_cases ht : c<2*r.half
    · exact Memory.readback api project op r back s call.output.base call.memory _ h c ha hh ht
    · rw [Memory.tail_readback api project op r back s call.output.base call.memory _ h c hh hc (by omega)]
      simp only [scalarResult,ht,↓reduceIte]
      rw [call.inputBinding _ h c ha hh hc,roundtrip,call.preparedInput _ h c ha hh hc]
  refine ⟨rfl,call.outputShape,value,?_,?_⟩
  · intro b h m c hb hh hm hc k
    rw [api.read_bytes]
    exact congrArg (fun w => api.encode w k) (value b h m c hb hh hm hc)
  · intro b hb outside
    change api.byte (Memory.run api project op r back s call.output.base call.memory) b=_
    rw [Memory.byte_frame api project op r back s call.output.base call.memory b outside]
    exact call.frame b hb

theorem no_launch_return {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : Operations α) (r : Request) (back : Bool) (s : State α)
    (call : Preparation api embed r s) (h : r.size=0 ∨ r.half=0) :
    entry api embed project op r back s call=⟨call.memory,call.output⟩ := by
  simp [entry,Memory.run,Memory.no_launch_events project op r back s call.output.base h,Memory.runWrites]

#specNonCircular Preparation avoiding [entry,Memory.run,FP.Structural.exec]
#axiomsClean whole
#axiomsClean no_launch_return
end
end VeriTile.Bench.Repairs.RopeEmbedding.Native
