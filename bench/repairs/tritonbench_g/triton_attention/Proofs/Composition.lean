import bench.repairs.tritonbench_g.triton_attention.Proofs.BackwardReturn

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native Memory HostSeparate
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

/-- Re-reading the physical forward stores yields exactly the saved logical
snapshot. Input allocations are preserved by fresh output allocation; no
post-kernel value premise is assumed. -/
theorem forward_live_word (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (actual : Instructions) (owners : Owners)
    (before : ByteMemory) (scale : Word)
    (hp : ∀ r∈forwardLive,pointerChecks g (objects.forwardObjects.complete.tensor r)=Bool.true)
    (allocation : ForwardAllocation objects.forwardObjects.complete.tensor)
    (r : Buffer) (hr : r∈forwardLive) (n : Nat) (hn : n<bufferSize (checkedShape g hs) r) :
    callerWords objects.tensor
        (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale) r n=
      Native.forward (checkedShape g hs) (Native.ops actual) owners
        (callerWords objects.forwardObjects.complete.tensor before) scale r n := by
  have live (r : Buffer) (hr : r∈forwardLive) : objects.forwardObjects.complete.tensor r=objects.tensor r :=
    complete_live objects.forwardObjects r hr
  have hw : ∀ r∈forwardWrites,pointerChecks g (objects.forwardObjects.complete.tensor r)=Bool.true := by
    intro r hr
    apply hp r
    simp only [forwardWrites,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl <;> simp [forwardLive]
  by_cases hwritten : r∈forwardWrites
  · unfold callerWords
    rw [←live r hr]
    exact finish_word g hs device objects.forwardObjects.complete forwardWrites (by simp [forwardWrites]) hw
      allocation.writes _ before r hwritten ⟨n,hn⟩
  · have input : r∈[Buffer.q,.k,.v] := by
      simp only [forwardLive,List.mem_cons,List.not_mem_nil,or_false] at hr
      simp only [forwardWrites,List.mem_cons,List.not_mem_nil,or_false,not_or] at hwritten
      simp only [List.mem_cons,List.not_mem_nil,or_false]
      tauto
    have hnwritten : ¬forwardWritten (checkedShape g hs) r n := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at input
      rcases input with rfl|rfl|rfl <;> simp [forwardWritten]
    rw [HostSeparate.forward_frame _ _ _ _ _ _ _ hnwritten]
    unfold callerWords
    rw [←live r hr]
    apply read_preserved g _ (hp r hr) (objects.forwardObjects.complete.width r (by
      intro he;subst r;simp [forwardLive] at hr))
    · intro location hlo hhi
      exact finish_preserves_allocation g hs device objects.forwardObjects.complete forwardWrites
        (by simp [forwardWrites]) hw _ before _ (fun s h=>allocation.inputs s h r input) location hlo hhi
    · rw [objects.forwardObjects.complete.count];exact hn

theorem forward_saved_match (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (actual : Instructions) (owners : Owners)
    (before : ByteMemory) (scale : Word)
    (hp : ∀ r∈forwardLive,pointerChecks g (objects.forwardObjects.complete.tensor r)=Bool.true)
    (allocation : ForwardAllocation objects.forwardObjects.complete.tensor) :
    SavedMatch (checkedShape g hs)
      (Native.forward (checkedShape g hs) (Native.ops actual) owners
        (callerWords objects.forwardObjects.complete.tensor before) scale)
      (callerWords objects.tensor
        (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale)) := by
  constructor
  all_goals first
    | (funext h i e;exact forward_live_word g hs device objects actual owners before scale hp allocation _
        (by simp [forwardLive]) _ (offset_lt _ _))
    | (funext h i;exact forward_live_word g hs device objects actual owners before scale hp allocation _
        (by simp [forwardLive]) _ (rowOffset_lt _ _ _))

theorem snapshot_saved (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (snapshot current : Memory Word) (gradient : Nat→Word) (saved : SavedMatch c snapshot current) :
    backwardSnapshot c p owners current gradient=backwardSnapshot c p owners snapshot gradient := by
  simp only [backwardSnapshot,Native.saved,saved.q,saved.k,saved.v,saved.o,saved.l,saved.m]

/-- The complete native references depend on the original Q/K/V bytes and the
cotangent actually supplied at the later backward point. Forward has no
cotangent parameter. The graph retains each native rounding/owner operation. -/
def WholeValues (c : Shape) (forwardObjects objects : Buffer→Tensor) (incoming : Tensor)
    (reference : Instructions) (owners : Owners) (initial forwardMemory finalMemory : ByteMemory) (scale : Word) : Prop :=
  let original:=callerWords forwardObjects initial
  let gradient:=gradientWord c incoming forwardMemory
  let backward:=wholeBackwardInput c (Native.ops reference) owners (Native.input c (attachGradient gradient original)) scale
  ∀ i : Coord c,
    logicalRead c (forwardObjects .o) finalMemory i=
      (forwardReference c (Native.ops reference) owners (primalInput c original) scale).y
        i.1.val i.2.1.val i.2.2.val ∧
    logicalRead c (objects .dq) finalMemory i=dqReference c (Native.ops reference) backward scale c.blocks
      i.1.val i.2.1.val i.2.2.val ∧
    logicalRead c (objects .dk) finalMemory i=dkReference c (Native.ops reference) backward scale
      i.1.val i.2.1.val i.2.2.val ∧
    logicalRead c (objects .dv) finalMemory i=dvReference c (Native.ops reference) backward scale
      i.1.val i.2.1.val i.2.2.val

theorem composed_values (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor)
    (actual reference : Instructions) (owners : Owners) (before finalMemory : ByteMemory) (scale : Word)
    (hp : ∀ r∈forwardLive,pointerChecks g (objects.forwardObjects.complete.tensor r)=Bool.true)
    (allocation : ForwardAllocation objects.forwardObjects.complete.tensor)
    (numeric : LocalConformance actual reference)
    (output : ∀ i : Coord (checkedShape g hs),
      logicalRead (checkedShape g hs) (objects.forwardObjects.complete.tensor .o) finalMemory i=
      logicalRead (checkedShape g hs) (objects.forwardObjects.complete.tensor .o)
        (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale) i)
    (fv : ForwardValues (checkedShape g hs) objects.forwardObjects.complete.tensor reference owners before
      (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale) scale)
    (bv : BackwardValues (checkedShape g hs) objects.tensor incoming reference owners
      (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale) finalMemory scale) :
    WholeValues (checkedShape g hs) objects.forwardObjects.complete.tensor objects.tensor incoming reference owners before
      (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale) finalMemory scale := by
  have he:=local_conformance_eq actual reference numeric
  subst actual
  have saved:=forward_saved_match g hs device objects reference owners before scale hp allocation
  have hb:=snapshot_saved (checkedShape g hs) (Native.ops reference) owners _ _
    (gradientWord (checkedShape g hs) incoming
      (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor reference owners before scale)) saved
  simp only [BackwardValues,hb,backwardSnapshot_forward] at bv
  intro i
  exact ⟨(output i).trans (fv.1 i),bv i⟩

#axiomsClean forward_live_word
#axiomsClean forward_saved_match
#axiomsClean snapshot_saved
#axiomsClean composed_values
#specNonCircular WholeValues avoiding [forwardBytes,backwardBytes,forwardCall,backwardCall,composed_values]
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
