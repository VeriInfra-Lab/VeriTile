import bench.repairs.tritonbench_g.triton_attention.Proofs.ByteViews

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native Memory HostSeparate
noncomputable section
open scoped Classical
set_option maxHeartbeats 2500000

/-- Ordinary observed allocation separation: the three fresh output/scratch
allocations are mutually disjoint and disjoint from all caller input storage. -/
structure ForwardAllocation (objects : Buffer→Tensor) : Prop where
  writes : SeparateWrites objects forwardWrites
  inputs : ∀ r∈forwardWrites,∀ input∈[Buffer.q,.k,.v],Disjoint (objects r) (objects input)

def forwardChecks (g : Geometry) (objects : Buffer→Tensor) (scale : ScalarABI) :=
  inputChecks g (objects .q) (objects .k) (objects .v) scale &&
    forwardLive.all (fun r=>pointerChecks g (objects r))

theorem forward_pointers (g : Geometry) (objects : Buffer→Tensor) (scale : ScalarABI)
    (checks : forwardChecks g objects scale=Bool.true) :
    ∀ r∈forwardLive,pointerChecks g (objects r)=Bool.true := by
  simp only [forwardChecks,Bool.and_eq_true,List.all_eq_true] at checks
  exact checks.2

structure ForwardReturn where
  output : Tensor
  saved : List Tensor
  rawScale : RawScalar
  launchDevice : Nat
  restoredDevice : Nat
  memory : ByteMemory

def forwardBytes (c : Shape) (objects : Buffer→Tensor) (actual : Instructions) (owners : Owners)
    (before : ByteMemory) (scale : Word) : ByteMemory :=
  finishBytes c objects forwardWrites
    (Native.forward c (Native.ops actual) owners (callerWords objects before) scale) before

def forwardReturn (c : Shape) (device priorDevice : Nat) (objects : Buffer→Tensor)
    (actual : Instructions) (owners : Owners) (before : ByteMemory) (scale : ScalarABI) : ForwardReturn :=
  ⟨objects .o,forwardLive.map objects,scale.raw,device,priorDevice,
    forwardBytes c objects actual owners before scale.word⟩

/-- The source checks, its actual output objects, owning-device launch context,
and the three executed stores. Runtime delivery is ordinary successful launch
interpretation; the definition contains no supplied result/safety conclusion. -/
def forwardCall (g : Geometry) (hs : shapeChecks g=Bool.true) (device priorDevice : Nat)
    (objects : ForwardObjects (checkedShape g hs) device) (actual : Instructions) (owners : Owners)
    (before : ByteMemory) (scale : ScalarABI) : Option ForwardReturn :=
  if forwardChecks g objects.complete.tensor scale then
    some (forwardReturn (checkedShape g hs) device priorDevice objects.complete.tensor actual owners before scale)
  else none

def ForwardValues (c : Shape) (objects : Buffer→Tensor) (reference : Instructions) (owners : Owners)
    (before after : ByteMemory) (scale : Word) : Prop :=
  (∀ i : Coord c,logicalRead c (objects .o) after i=
    (forwardReference c (Native.ops reference) owners (primalInput c (callerWords objects before)) scale).y
      i.1.val i.2.1.val i.2.2.val) ∧
  (∀ i : Fin c.groups × Fin c.rows,
    rowRead (objects .l) after i=
      (forwardReference c (Native.ops reference) owners (primalInput c (callerWords objects before)) scale).l
        i.1.val i.2.val ∧
    rowRead (objects .m) after i=
      (forwardReference c (Native.ops reference) owners (primalInput c (callerWords objects before)) scale).m
        i.1.val i.2.val)

theorem forward_values (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (actual reference : Instructions) (owners : Owners)
    (before : ByteMemory) (scale : Word)
    (hp : ∀ r∈forwardWrites,pointerChecks g (objects.tensor r)=Bool.true)
    (allocation : SeparateWrites objects.tensor forwardWrites)
    (numeric : LocalConformance actual reference) :
    ForwardValues (checkedShape g hs) objects.tensor reference owners before
      (forwardBytes (checkedShape g hs) objects.tensor actual owners before scale) scale := by
  rw [local_conformance_eq actual reference numeric]
  have hv:=forward_correct (checkedShape g hs) reference owners (callerWords objects.tensor before) scale
  constructor
  · intro i
    rw [logical_read _ device objects _ .o (by decide) (by decide)]
    change readWord (finishBytes _ _ _ _ _) _=_
    rw [finish_word g hs device objects forwardWrites (by simp [forwardWrites]) hp allocation _ before .o
      (by simp [forwardWrites]) ⟨offset _ i,offset_lt _ i⟩]
    exact hv.1 i
  · intro i
    constructor
    · rw [row_read _ device objects _ .l (by decide) (by decide)]
      change readWord (finishBytes _ _ _ _ _) _=_
      rw [finish_word g hs device objects forwardWrites (by simp [forwardWrites]) hp allocation _ before .l
        (by simp [forwardWrites]) ⟨rowAddr _ i,rowOffset_lt _ _ _⟩]
      exact (hv.2.1 i).1
    · rw [row_read _ device objects _ .m (by decide) (by decide)]
      change readWord (finishBytes _ _ _ _ _) _=_
      rw [finish_word g hs device objects forwardWrites (by simp [forwardWrites]) hp allocation _ before .m
        (by simp [forwardWrites]) ⟨rowAddr _ i,rowOffset_lt _ _ _⟩]
      exact (hv.2.1 i).2

/-- Separate checked forward endpoint. No future cotangent is an argument.
The exact result Tensor, saved objects, raw scalar, and device restoration are
part of the conclusion; every byte outside the three write extents is framed. -/
theorem forward_native (g : Geometry) (hs : shapeChecks g=Bool.true) (device priorDevice : Nat)
    (objects : ForwardObjects (checkedShape g hs) device) (actual reference : Instructions) (owners : Owners)
    (before : ByteMemory) (scale : ScalarABI)
    (checks : forwardChecks g objects.complete.tensor scale=Bool.true)
    (allocation : ForwardAllocation objects.complete.tensor)
    (numeric : LocalConformance actual reference) :
    ∃ result,forwardCall g hs device priorDevice objects actual owners before scale=some result ∧
      result.output=objects.tensor .o ∧ result.saved=forwardLive.map objects.tensor ∧
      result.rawScale=scale.raw ∧ result.launchDevice=device ∧ result.restoredDevice=priorDevice ∧
      ForwardValues (checkedShape g hs) objects.complete.tensor reference owners before result.memory scale.word ∧
      (∀ location : Fin (2^64),
        (∀ i : ByteIndex (checkedShape g hs) forwardWrites,
          location.val≠byteAddress _ objects.complete.tensor forwardWrites i) → result.memory location=before location) ∧
      (∀ input∈[Buffer.q,.k,.v],∀ location : Fin (2^64),
        (objects.tensor input).storageBase≤location.val →
        location.val<(objects.tensor input).storageBase+(objects.tensor input).storageBytes →
        result.memory location=before location) := by
  have hp:=forward_pointers g objects.complete.tensor scale checks
  have writesLive : ∀ r∈forwardWrites,r∈forwardLive := by
    intro r hr;simpa [forwardWrites,forwardLive] using Or.inr (Or.inr (Or.inr hr))
  have hwp : ∀ r∈forwardWrites,pointerChecks g (objects.complete.tensor r)=Bool.true :=
    fun r hr=>hp r (writesLive r hr)
  refine ⟨forwardReturn _ device priorDevice objects.complete.tensor actual owners before scale,?_,?_,?_,rfl,rfl,rfl,
    forward_values g hs device objects.complete actual reference owners before scale.word hwp allocation.writes numeric,?_,?_⟩
  · simp only [forwardCall,checks,↓reduceIte]
  · exact complete_live objects .o (by simp [forwardLive])
  · apply List.map_congr_left
    intro r hr
    exact complete_live objects r hr
  · intro location outside
    exact finish_frame _ _ _ _ _ location outside
  · intro input hi location hlo hhi
    have hl : input∈forwardLive := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hi
      rcases hi with rfl|rfl|rfl <;> simp [forwardLive]
    rw [←complete_live objects input hl] at hlo hhi
    exact finish_preserves_allocation g hs device objects.complete forwardWrites (by simp [forwardWrites]) hwp
      _ before (objects.complete.tensor input) (fun r hr=>allocation.inputs r hr input hi) location hlo hhi

#axiomsClean forward_pointers
#axiomsClean forward_values
#axiomsClean forward_native
#specNonCircular ForwardValues avoiding [forwardBytes,forwardCall,forward_native]
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
