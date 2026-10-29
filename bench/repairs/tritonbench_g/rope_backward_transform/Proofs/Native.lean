import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Memory
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
noncomputable section
set_option maxHeartbeats 3000000

structure Tensor where
  (handle storage device base allocationBase allocationEnd : Nat)
  (shape strides : List Nat)
  dtype : String
  (cuda materialized : Bool)
  elementBytes : Nat
  (strided contiguous : Bool)

def logicalAddress (t : Tensor) (b h time c : Nat) :=
  t.base+4*(b*t.strides[0]!+h*t.strides[1]!+time*t.strides[2]!+c*t.strides[3]!)
def logicalRead (api : API β Store Byte) (store : Store) (t : Tensor) (b h time c : Nat) :=
  api.read store (logicalAddress t b h time c)
def tableAddress (t : Tensor) (time j : Nat) := t.base+4*(time*t.strides[0]!+j*t.strides[1]!)
def span (rows half rowStride colStride : Nat) :=
  if rows=0 ∨ half=0 then 0 else (rows-1)*rowStride+(half-1)*colStride+1

def canReuse (r : Request) (key : Bool) : Prop := r.size key=0 ∨ r.sequence≤1 ∨ r.heads key≤1
instance (r : Request) (key : Bool) : Decidable (canReuse r key) := inferInstanceAs (Decidable (_∨_∨_))

/-- Ordinary transpose/contiguous preparation and returned transpose views.
Content premises mention only input copies and reads, never a rotated result.
The output may share the input storage exactly when contiguous() reuses it. -/
structure Preparation {α β Store Byte : Type} [Inhabited α] (api : API β Store Byte)
    (embed : β→α) (r : Request) (initial : Source.Memory α) where
  (before packed : Store)
  (input output : Bool→Tensor)
  (cosine sine : Tensor)
  caller : Nat→Prop
  callerBinding : ∀ b,caller b ↔
    (∃ k,(input k).allocationBase≤b ∧ b<(input k).allocationEnd) ∨
    (cosine.allocationBase≤b ∧ b<cosine.allocationEnd) ∨
    (sine.allocationBase≤b ∧ b<sine.allocationEnd)
  reused : Bool→Bool
  inputShape : ∀ k,(input k).shape=[r.batches,r.heads k,r.sequence,r.dimension]
  outputShape : ∀ k,(output k).shape=[r.batches,r.heads k,r.sequence,r.dimension]
  cosineShape : cosine.shape=[r.cosRows,r.half]
  sineShape : sine.shape=[r.sinRows,r.half]
  tableRows : r.sequence≤r.cosRows ∧ r.sequence≤r.sinRows
  inputType : ∀ k,(input k).dtype="float32" ∧ (input k).elementBytes=4
  outputType : ∀ k,(output k).dtype="float32" ∧ (output k).elementBytes=4
  cosineType : cosine.dtype="float32" ∧ cosine.elementBytes=4
  sineType : sine.dtype="float32" ∧ sine.elementBytes=4
  resident : ∀ k,(input k).cuda=Bool.true ∧ (output k).cuda=Bool.true ∧ cosine.cuda=Bool.true ∧ sine.cuda=Bool.true
  materialized : ∀ k,(input k).materialized=Bool.true ∧ (output k).materialized=Bool.true ∧ cosine.materialized=Bool.true ∧ sine.materialized=Bool.true
  devices : ∀ k,(input k).device=(input Bool.false).device ∧ (output k).device=(input Bool.false).device ∧
    cosine.device=(input Bool.false).device ∧ sine.device=(input Bool.false).device
  coefficientStrides : cosine.strides=[r.cosRowStride,r.cosColStride] ∧ sine.strides=[r.sinRowStride,r.sinColStride]
  inputAddress : ∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
    logicalAddress (input k) b h time c=(input k).base+4*r.external k b h time c
  outputAddress : ∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
    logicalAddress (output k) b h time c=(output k).base+4*r.offset k (b*r.sequence+time) h c
  reuseCondition : ∀ k,reused k=decide (canReuse r k)
  reuseIdentity : ∀ k,reused k=Bool.true → (output k).storage=(input k).storage ∧ (output k).base=(input k).base
  copiedFresh : ∀ k,reused k=Bool.false → ∀ b,caller b → b<(output k).base ∨ (output k).base+4*r.size k≤b
  nativeStorageSeparation : (input Bool.false).storage≠(input Bool.true).storage ∧
    (∀ k,(input k).storage≠cosine.storage ∧ (input k).storage≠sine.storage)
  outputSeparate : Memory.Separated r (fun k => (output k).base)
  coefficientSeparate : ∀ k,
    ((output k).base+4*r.size k≤cosine.base ∨ cosine.base+4*span r.cosRows r.half r.cosRowStride r.cosColStride≤(output k).base) ∧
    ((output k).base+4*r.size k≤sine.base ∨ sine.base+4*span r.sinRows r.half r.sinRowStride r.sinColStride≤(output k).base)
  copiedInputSeparate : ∀ k k',reused k=Bool.false →
    (output k').base+4*r.size k'≤(input k).base ∨ (input k).base+4*r.size k≤(output k').base
  inputBinding : ∀ k a h c,a<r.rows → h<r.heads k → c<r.dimension →
    (initial (region k) (r.offset k a h c)).read .fp32=
      embed (api.read before ((input k).base+4*r.external k (a/r.sequence) h (a%r.sequence) c))
  packedBinding : ∀ k a h c,a<r.rows → h<r.heads k → c<r.dimension →
    api.read packed ((output k).base+4*r.offset k a h c)=
      api.read before ((input k).base+4*r.external k (a/r.sequence) h (a%r.sequence) c)
  cosineBinding : ∀ time j,time<r.sequence → j<r.half →
    (initial "C" (time*r.cosRowStride+j*r.cosColStride)).read .fp32=embed (api.read before (tableAddress cosine time j))
  sineBinding : ∀ time j,time<r.sequence → j<r.half →
    (initial "S" (time*r.sinRowStride+j*r.sinColStride)).read .fp32=embed (api.read before (tableAddress sine time j))
  preservedCosine : ∀ time j,time<r.sequence → j<r.half → api.read packed (tableAddress cosine time j)=api.read before (tableAddress cosine time j)
  preservedSine : ∀ time j,time<r.sequence → j<r.half → api.read packed (tableAddress sine time j)=api.read before (tableAddress sine time j)
  inputFits : ∀ k,(input k).base+4*r.size k≤(input k).allocationEnd ∧ (input k).allocationEnd≤2^64
  outputFits : ∀ k,(output k).base+4*r.size k≤(output k).allocationEnd ∧ (output k).allocationEnd≤2^64
  cosineFits : cosine.base+4*span r.cosRows r.half r.cosRowStride r.cosColStride≤cosine.allocationEnd ∧ cosine.allocationEnd≤2^64
  sineFits : sine.base+4*span r.sinRows r.half r.sinRowStride r.sinColStride≤sine.allocationEnd ∧ sine.allocationEnd≤2^64
  frame : ∀ b,caller b → api.byte packed b=api.byte before b
  inputOwned : ∀ k b,(input k).base≤b → b<(input k).base+4*r.size k → caller b

structure Returned (Store : Type) where
  memory : Store
  (q k : Tensor)

def launchRows (r : Request) : List Nat :=
  if r.rows=0 ∨ r.half=0 ∨ (r.qHeads=0 ∧ r.kHeads=0) then [] else List.range r.rows

theorem launchRows_valid (r : Request) (a : Nat) (ha : a∈launchRows r) : a<r.rows := by
  unfold launchRows at ha
  split at ha
  · simp at ha
  · exact List.mem_range.mp ha

theorem launchRows_present (r : Request) (k : Bool) (a h c : Nat)
    (ha : a<r.rows) (hh : h<r.heads k) (hc : c<2*r.half) : a∈launchRows r := by
  have nz : ¬(r.rows=0 ∨ r.half=0 ∨ (r.qHeads=0 ∧ r.kHeads=0)) := by
    intro hz
    rcases hz with hz | hz | hz
    · omega
    · omega
    · cases k <;> simp_all [Request.heads]
  simp [launchRows,nz,ha]

theorem launchRows_nodup (r : Request) : (launchRows r).Nodup := by
  unfold launchRows
  split
  · exact List.nodup_nil
  · exact List.nodup_range

def entry {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Preparation api embed r m) : Returned Store :=
  ⟨Memory.run api project op r m (fun k => (call.output k).base) (launchRows r) call.packed,
    call.output Bool.false,call.output Bool.true⟩
def returned (out : Returned Store) (key : Bool) := if key then out.k else out.q

/-- Input-only integer domain for the actual i32 source specialization, including
formed masked pointers, padded lanes, both halves and byte address arithmetic. -/
def IndexABI {α β Store Byte : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α)
    (r : Request) (m : Source.Memory α) (call : Preparation api embed r m) : Prop :=
  r.rows≠0 → r.half≠0 → ¬(r.qHeads=0 ∧ r.kHeads=0) →
  r.batches<2^31 ∧ r.rows<2^31 ∧ r.sequence<2^31 ∧ r.dimension<2^31 ∧ r.paddedDim≤65536 ∧
  r.cosRowStride<2^31 ∧ r.cosColStride<2^31 ∧ r.sinRowStride<2^31 ∧ r.sinColStride<2^31 ∧
  ∀ k,r.heads k<2^31 ∧ r.heads k*r.dimension<2^31 ∧ r.paddedHeads k*r.paddedDim≤65536 ∧
    ∀ row<r.rows,∀ h<r.paddedHeads k,∀ j<r.paddedDim/2,
      row*(r.heads k*r.dimension)+h*r.dimension+j+r.half<2^31 ∧
      row%r.sequence*r.cosRowStride+j*r.cosColStride<2^31 ∧
      row%r.sequence*r.sinRowStride+j*r.sinColStride<2^31 ∧
      (call.output k).base+4*(row*(r.heads k*r.dimension)+h*r.dimension+j+r.half)<2^64 ∧
      call.cosine.base+4*(row%r.sequence*r.cosRowStride+j*r.cosColStride)<2^64 ∧
      call.sine.base+4*(row%r.sequence*r.sinRowStride+j*r.sinColStride)<2^64

theorem whole {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Preparation api embed r m)
    (roundtrip : ∀ w,project (embed w)=w) :
    (∀ k,returned (entry api embed project op r m call) k=call.output k) ∧
    (∀ k,(returned (entry api embed project op r m call) k).shape=[r.batches,r.heads k,r.sequence,r.dimension]) ∧
    (∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
      logicalRead api (entry api embed project op r m call).memory
        (returned (entry api embed project op r m call) k) b h time c=
          project (scalarResult op r m k (b*r.sequence+time) h c)) ∧
    (∀ b,call.caller b → Memory.Outside r (fun k => (call.output k).base) (launchRows r) b →
      api.byte (entry api embed project op r m call).memory b=api.byte call.before b) := by
  have object (k : Bool) : returned (entry api embed project op r m call) k=call.output k := by cases k <;> rfl
  refine ⟨object,fun k => by rw [object k]; exact call.outputShape k,?_,?_⟩
  · intro k b h time c hb hh ht hc
    rw [object k]
    change api.read (Memory.run api project op r m _ _ _) (logicalAddress (call.output k) b h time c)=_
    rw [call.outputAddress k b h time c hb hh ht hc]
    have ha := row_bound r b time hb ht
    by_cases active : c<2*r.half
    · exact Memory.readback api project op r m _ _ _ call.outputSeparate
        (launchRows_valid r) k _ h c (launchRows_present r k _ h c ha hh active) hh active
    · rw [Memory.coordinate_frame api project op r m _ _ _ call.outputSeparate
        (launchRows_valid r) k _ h c ha hh hc (Or.inr (by omega))]
      simp only [scalarResult,active,↓reduceIte]
      rw [call.inputBinding k _ h c ha hh hc,roundtrip,call.packedBinding k _ h c ha hh hc]
  · intro b hb hout
    change api.byte (Memory.run api project op r m _ _ _) b=_
    rw [Memory.byte_frame api project op r m _ _ _ b hout]
    exact call.frame b hb

theorem no_launch_return {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Preparation api embed r m)
    (h : r.rows=0 ∨ r.half=0 ∨ (r.qHeads=0 ∧ r.kHeads=0)) :
    entry api embed project op r m call=⟨call.packed,call.output Bool.false,call.output Bool.true⟩ := by
  have hn : launchRows r=[] := by simp [launchRows,h]
  simp [entry,Memory.run,Memory.no_launch_events project op r m _ _ (Or.inl hn),RopeEmbedding.Memory.runWrites]

#specNonCircular Preparation avoiding [entry,Memory.run,scalarResult,FP.Structural.exec]
#axiomsClean whole
#axiomsClean no_launch_return
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.Native
