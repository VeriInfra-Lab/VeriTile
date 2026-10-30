import bench.repairs.tritonbench_g.triton_attention.Proofs.ForwardReturn

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native Memory HostSeparate
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

def backwardLive : List Buffer := Buffer.g::forwardLive++backwardWrites

/-- Ordinary allocation outcomes of the retained contiguous operation and the
five fresh backward allocations, not supplied kernel-output conclusions. -/
structure BackwardAllocation (objects : Buffer→Tensor) (incoming : Tensor) : Prop where
  copy : CopyAllocation objects incoming
  writes : SeparateWrites objects backwardWrites
  inputs : ∀ r∈backwardWrites,∀ t∈incoming::(Buffer.g::forwardLive).map objects,Disjoint (objects r) t

def backwardChecks (g : Geometry) (device : Nat) (objects : Buffer→Tensor) (incoming : Tensor) :=
  gradientChecks g device incoming && backwardLive.all (fun r=>pointerChecks g (objects r))

theorem backward_pointers (g : Geometry) (device : Nat) (objects : Buffer→Tensor) (incoming : Tensor)
    (checks : backwardChecks g device objects incoming=Bool.true) :
    ∀ r∈backwardLive,pointerChecks g (objects r)=Bool.true := by
  simp only [backwardChecks,Bool.and_eq_true,List.all_eq_true] at checks
  exact checks.2

structure BackwardReturn where
  dq : Tensor
  dk : Tensor
  dv : Tensor
  scaleGradient : Option Tensor
  rawScale : RawScalar
  launchDevice : Nat
  restoredDevice : Nat
  memory : ByteMemory

def copiedWords (c : Shape) (objects : Buffer→Tensor) (incoming : Tensor) (before : ByteMemory) :=
  callerWords objects (copyGradient c objects incoming before)

/-- The exact original order: gradient contiguous, zerosDQ, preprocessing,
then the current-memory backward loops. -/
def backwardWords (c : Shape) (objects : Buffer→Tensor) (incoming : Tensor)
    (actual : Instructions) (owners : Owners) (before : ByteMemory) (scale : Word) : Memory Word :=
  HostSeparate.backwardRun c (Native.ops actual) owners (copiedWords c objects incoming before)
    (copiedWords c objects incoming before .g) scale

def backwardBytes (c : Shape) (objects : Buffer→Tensor) (incoming : Tensor)
    (actual : Instructions) (owners : Owners) (before : ByteMemory) (scale : Word) : ByteMemory :=
  finishBytes c objects backwardWrites (backwardWords c objects incoming actual owners before scale)
    (copyGradient c objects incoming before)

def backwardReturn (c : Shape) (device priorDevice : Nat) (objects : Buffer→Tensor) (incoming : Tensor)
    (actual : Instructions) (owners : Owners) (before : ByteMemory) (scale : ScalarABI) : BackwardReturn :=
  ⟨objects .dq,objects .dk,objects .dv,none,scale.raw,device,priorDevice,
    backwardBytes c objects incoming actual owners before scale.word⟩

def backwardCall (g : Geometry) (hs : shapeChecks g=Bool.true) (device priorDevice : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor)
    (actual : Instructions) (owners : Owners) (before : ByteMemory) (scale : ScalarABI) : Option BackwardReturn :=
  if backwardChecks g device objects.tensor incoming then
    some (backwardReturn (checkedShape g hs) device priorDevice objects.tensor incoming actual owners before scale)
  else none

theorem copy_gradient_view (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor) (before : ByteMemory)
    (hp : pointerChecks g (objects.tensor .g)=Bool.true) :
    gradientView (checkedShape g hs) (copiedWords (checkedShape g hs) objects.tensor incoming before .g)=
      gradientView (checkedShape g hs) (gradientWord (checkedShape g hs) incoming before) := by
  funext h i e
  exact copy_read g hs device objects incoming before hp ⟨offset _ _,offset_lt _ _⟩

def BackwardValues (c : Shape) (objects : Buffer→Tensor) (incoming : Tensor)
    (reference : Instructions) (owners : Owners) (before after : ByteMemory) (scale : Word) : Prop :=
  let snapshot:=backwardSnapshot c (Native.ops reference) owners (callerWords objects before) (gradientWord c incoming before)
  ∀ i : Coord c,
    logicalRead c (objects .dq) after i=dqReference c (Native.ops reference) snapshot scale c.blocks
      i.1.val i.2.1.val i.2.2.val ∧
    logicalRead c (objects .dk) after i=dkReference c (Native.ops reference) snapshot scale
      i.1.val i.2.1.val i.2.2.val ∧
    logicalRead c (objects .dv) after i=dvReference c (Native.ops reference) snapshot scale
      i.1.val i.2.1.val i.2.2.val

theorem backward_values (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor)
    (actual reference : Instructions) (owners : Owners) (before : ByteMemory) (scale : Word)
    (hp : ∀ r∈backwardLive,pointerChecks g (objects.tensor r)=Bool.true)
    (allocation : BackwardAllocation objects.tensor incoming)
    (numeric : LocalConformance actual reference) :
    BackwardValues (checkedShape g hs) objects.tensor incoming reference owners before
      (backwardBytes (checkedShape g hs) objects.tensor incoming actual owners before scale) scale := by
  rw [local_conformance_eq actual reference numeric]
  have hcopy : ∀ r∈Buffer.g::forwardLive,pointerChecks g (objects.tensor r)=Bool.true := by
    intro r hr;apply hp r;simpa only [backwardLive,List.mem_append] using Or.inl hr
  have hwrites : ∀ r∈backwardWrites,pointerChecks g (objects.tensor r)=Bool.true := by
    intro r hr;apply hp r;simpa only [backwardLive,List.mem_append] using Or.inr hr
  have saved:=copy_saved_match g hs device objects incoming before hcopy allocation.copy
  have hv:=(HostSeparate.backward_correct (checkedShape g hs) reference owners
    (callerWords objects.tensor before) (copiedWords (checkedShape g hs) objects.tensor incoming before)
    (copiedWords (checkedShape g hs) objects.tensor incoming before .g) scale saved).1
  have hg:=copy_gradient_view g hs device objects incoming before (hcopy .g (by simp))
  have hb : backwardSnapshot (checkedShape g hs) (Native.ops reference) owners (callerWords objects.tensor before)
      (copiedWords (checkedShape g hs) objects.tensor incoming before .g)=
    backwardSnapshot (checkedShape g hs) (Native.ops reference) owners (callerWords objects.tensor before)
      (gradientWord (checkedShape g hs) incoming before) := by
    unfold backwardSnapshot
    rw [hg]
  simp only [hb] at hv
  intro i
  have readback (r : Buffer) (hr : r∈backwardWrites) (hrow : isRows r=Bool.false) :
      logicalRead (checkedShape g hs) (objects.tensor r)
        (backwardBytes (checkedShape g hs) objects.tensor incoming reference owners before scale) i=
      backwardWords (checkedShape g hs) objects.tensor incoming reference owners before scale r (offset _ i) := by
    have hne : r≠Buffer.other := by intro he;subst r;simp [backwardWrites] at hr
    rw [logical_read _ device objects _ r hne hrow]
    exact finish_word g hs device objects backwardWrites (by simp [backwardWrites]) hwrites allocation.writes
      _ _ r hr ⟨offset _ i,by cases r <;> simp_all [bufferSize,isRows,offset_lt]⟩
  change _ ∧ _ ∧ _
  rw [readback .dq (by simp [backwardWrites]) (by decide),readback .dk (by simp [backwardWrites]) (by decide),
    readback .dv (by simp [backwardWrites]) (by decide)]
  exact hv i

/-- Separate later backward endpoint, using the actual incoming cotangent and
actual saved byte views. It returns DQ/DK/DV/None and frames the complete global
memory outside the copy and backward writes; all caller/saved allocations are
preserved even when the original contiguous operation returns its input. -/
theorem backward_native (g : Geometry) (hs : shapeChecks g=Bool.true) (device priorDevice : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor)
    (actual reference : Instructions) (owners : Owners) (before : ByteMemory) (scale : ScalarABI)
    (checks : backwardChecks g device objects.tensor incoming=Bool.true)
    (allocation : BackwardAllocation objects.tensor incoming)
    (numeric : LocalConformance actual reference) :
    ∃ result,backwardCall g hs device priorDevice objects incoming actual owners before scale=some result ∧
      result.dq=objects.tensor .dq ∧ result.dk=objects.tensor .dk ∧ result.dv=objects.tensor .dv ∧
      result.scaleGradient=none ∧ result.rawScale=scale.raw ∧
      result.launchDevice=device ∧ result.restoredDevice=priorDevice ∧
      BackwardValues (checkedShape g hs) objects.tensor incoming reference owners before result.memory scale.word ∧
      (∀ i : ByteIndex (checkedShape g hs) backwardWrites,
        extendMemory result.memory (byteAddress _ objects.tensor backwardWrites i)=
          encode (backwardWords (checkedShape g hs) objects.tensor incoming actual owners before scale.word i.1.val i.2.1.val) i.2.2) ∧
      (∀ location : Fin (2^64),
        (∀ i : ByteIndex (checkedShape g hs) backwardWrites,location.val≠byteAddress _ objects.tensor backwardWrites i) →
        (∀ i : ByteIndex (checkedShape g hs) [.g],location.val≠byteAddress _ objects.tensor [.g] i) →
        result.memory location=before location) ∧
      (∀ t∈incoming::forwardLive.map objects.tensor,∀ location : Fin (2^64),
        t.storageBase≤location.val → location.val<t.storageBase+t.storageBytes → result.memory location=before location) := by
  have hp:=backward_pointers g device objects.tensor incoming checks
  have hcopy : ∀ r∈Buffer.g::forwardLive,pointerChecks g (objects.tensor r)=Bool.true := by
    intro r hr;apply hp r;simpa only [backwardLive,List.mem_append] using Or.inl hr
  have hwrites : ∀ r∈backwardWrites,pointerChecks g (objects.tensor r)=Bool.true := by
    intro r hr;apply hp r;simpa only [backwardLive,List.mem_append] using Or.inr hr
  refine ⟨backwardReturn _ device priorDevice objects.tensor incoming actual owners before scale,?_,rfl,rfl,rfl,rfl,rfl,rfl,rfl,
    backward_values g hs device objects incoming actual reference owners before scale.word hp allocation numeric,?_,?_,?_⟩
  · simp only [backwardCall,checks,↓reduceIte]
  · intro i
    exact finish_byte g hs device objects backwardWrites (by simp [backwardWrites]) hwrites allocation.writes _ _ i
  · intro location hout hcopyout
    change finishBytes _ _ _ _ _ location=before location
    rw [finish_frame _ _ _ _ _ location hout]
    exact finish_frame _ _ _ _ _ location hcopyout
  · intro t ht location hlo hhi
    have ht' : t∈incoming::(Buffer.g::forwardLive).map objects.tensor := by
      simp only [List.mem_cons,List.map_cons] at ht ⊢
      exact ht.elim Or.inl (fun h=>Or.inr (Or.inr h))
    change finishBytes _ _ _ _ _ location=before location
    rw [finish_preserves_allocation g hs device objects backwardWrites (by simp [backwardWrites]) hwrites _ _ t
      (fun r hr=>allocation.inputs r hr t ht')
      location hlo hhi]
    exact copy_preserves_inputs g hs device objects incoming before (hcopy .g (by simp)) allocation.copy t
      (by simpa only [forwardLive,List.map_cons,List.map_nil] using ht) location hlo hhi

#axiomsClean backward_pointers
#axiomsClean copy_gradient_view
#axiomsClean backward_values
#axiomsClean backward_native
#specNonCircular BackwardValues avoiding [backwardWords,backwardBytes,backwardCall,backward_native]
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
