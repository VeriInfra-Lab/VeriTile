import bench.repairs.tritonbench_g.triton_attention.Proofs.GradientCopy

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native Memory
noncomputable section
open scoped Classical
set_option maxHeartbeats 2500000

def forwardLive : List Buffer := [.q,.k,.v,.o,.l,.m]
def forwardWrites : List Buffer := [.o,.l,.m]
def backwardWrites : List Buffer := [.newDO,.delta,.dq,.dk,.dv]

/-- Only the six objects that actually exist at the forward endpoint. There is
no incoming gradient, backward allocation, or future-call premise. -/
structure ForwardObjects (c : Shape) (device : Nat) where
  tensor : Buffer→Tensor
  metadata : ∀ r∈forwardLive,(tensor r).observation=
    if isRows r then Host.rowTensor c device r else
      {Host.tensor c device r with strides:=(tensor r).observation.strides}
  packed : ∀ r∈forwardLive,isRows r=Bool.false → PackedStrides c (tensor r)
  width : ∀ r∈forwardLive,(tensor r).width=4
  count : ∀ r∈forwardLive,(tensor r).numel=bufferSize c r

/-- Unused roles have inert placeholders, not observations of future objects.
Only forwardLive is ever inspected by a forward source operation. -/
def ForwardObjects.complete {c : Shape} {device : Nat} (f : ForwardObjects c device) : Objects c device where
  tensor r := if r∈forwardLive then f.tensor r else
    { f.tensor .q with
      observation := if isRows r then Host.rowTensor c device r else Host.tensor c device r
      numel := bufferSize c r }
  metadata r _ := by
    by_cases hr : r∈forwardLive
    · simp only [if_pos hr];exact f.metadata r hr
    · simp only [if_neg hr]
      cases hrow : isRows r <;> simp [hrow,Host.tensor]
  packed r _ row := by
    by_cases hr : r∈forwardLive
    · simp only [if_pos hr];exact f.packed r hr row
    · simp only [if_neg hr]
      apply canonical_packed
      simp [row,Host.tensor]
  width r _ := by
    by_cases hr : r∈forwardLive
    · simp only [if_pos hr];exact f.width r hr
    · simp only [if_neg hr];exact f.width .q (by simp [forwardLive])
  count r := by
    by_cases hr : r∈forwardLive
    · simp only [if_pos hr];exact f.count r hr
    · simp only [if_neg hr]

@[simp] theorem complete_live {c : Shape} {device : Nat} (f : ForwardObjects c device)
    (r : Buffer) (hr : r∈forwardLive) : f.complete.tensor r=f.tensor r := by
  exact if_pos hr

def Objects.forwardObjects {c : Shape} {device : Nat} (objects : Objects c device) : ForwardObjects c device where
  tensor := objects.tensor
  metadata r hr := objects.metadata r (by intro he;subst r;simp [forwardLive] at hr)
  packed r hr := objects.packed r (by intro he;subst r;simp [forwardLive] at hr)
  width r hr := objects.width r (by intro he;subst r;simp [forwardLive] at hr)
  count r _ := objects.count r

/-- Read the actual Tensor through its observed four-dimensional strides. -/
def logicalRead (c : Shape) (t : Tensor) (memory : ByteMemory) (i : Coord c) : Word :=
  gradientWord c t memory (offset c i)

theorem logical_read (c : Shape) (device : Nat) (objects : Objects c device)
    (memory : ByteMemory) (r : Buffer) (hr : r≠.other) (row : isRows r=Bool.false) (i : Coord c) :
    logicalRead c (objects.tensor r) memory i=callerWords objects.tensor memory r (offset c i) := by
  unfold logicalRead gradientWord
  rw [gradient_offset_packed c _ (objects.packed r hr row) _ (offset_lt _ _)]
  rfl

/-- A row buffer is read using its actual two-dimensional stride observation. -/
def rowRead {c : Shape} (t : Tensor) (memory : ByteMemory) (i : Fin c.groups × Fin c.rows) : Word :=
  readWord memory (t.dataPointer+4*(i.1.val*t.observation.strides[0]?.getD 0+
    i.2.val*t.observation.strides[1]?.getD 0))

theorem row_read (c : Shape) (device : Nat) (objects : Objects c device)
    (memory : ByteMemory) (r : Buffer) (hr : r≠.other) (row : isRows r=Bool.true)
    (i : Fin c.groups × Fin c.rows) :
    rowRead (objects.tensor r) memory i=callerWords objects.tensor memory r (rowAddr c i) := by
  have hm:=congrArg Host.Tensor.strides (objects.metadata r hr)
  simp only [row,↓reduceIte,Host.rowTensor] at hm
  simp only [rowRead,hm,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Nat.mul_one,
    callerWords,rowAddr,rowOffset]

theorem read_preserved (g : Geometry) (t : Tensor) (hp : pointerChecks g t=Bool.true)
    (width : t.width=4) (before after : ByteMemory)
    (frame : ∀ location : Fin (2^64),t.storageBase≤location.val →
      location.val<t.storageBase+t.storageBytes → after location=before location)
    (n : Nat) (hn : n<t.numel) :
    readWord after (t.dataPointer+4*n)=readWord before (t.dataPointer+4*n) := by
  unfold readWord
  congr 1
  funext j
  by_cases hb : t.dataPointer+4*n+j.val<2^64
  · have h:=active_bytes g t hp n hn j.val (by rw [width];exact j.isLt)
    simp only [width] at h
    simp only [extendMemory,dif_pos hb]
    exact frame ⟨_,hb⟩ h.1 h.2
  · simp only [extendMemory,dif_neg hb]

theorem copy_saved_words (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor) (memory : ByteMemory)
    (hp : ∀ r∈Buffer.g::forwardLive,pointerChecks g (objects.tensor r)=Bool.true)
    (allocation : CopyAllocation objects.tensor incoming)
    (r : Buffer) (hr : r∈forwardLive) (n : Nat) (hn : n<bufferSize (checkedShape g hs) r) :
    callerWords objects.tensor (copyGradient (checkedShape g hs) objects.tensor incoming memory) r n=
      callerWords objects.tensor memory r n := by
  have hne : r≠Buffer.other := by intro he;subst r;simp [forwardLive] at hr
  apply read_preserved g (objects.tensor r) (hp r (by simp [hr])) (objects.width r hne)
  · intro location hlo hhi
    apply copy_preserves_inputs g hs device objects incoming memory (hp .g (by simp)) allocation
      (objects.tensor r) ?_ location hlo hhi
    simp only [forwardLive,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl|rfl <;> simp
  · rw [objects.count];exact hn

theorem copy_saved_match (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor) (memory : ByteMemory)
    (hp : ∀ r∈Buffer.g::forwardLive,pointerChecks g (objects.tensor r)=Bool.true)
    (allocation : CopyAllocation objects.tensor incoming) :
    HostSeparate.SavedMatch (checkedShape g hs) (callerWords objects.tensor memory)
      (callerWords objects.tensor (copyGradient (checkedShape g hs) objects.tensor incoming memory)) := by
  constructor
  all_goals first
    | (funext h i e;exact copy_saved_words g hs device objects incoming memory hp allocation _
        (by simp [forwardLive]) _ (offset_lt _ _))
    | (funext h i;exact copy_saved_words g hs device objects incoming memory hp allocation _
        (by simp [forwardLive]) _ (rowOffset_lt _ _ _))

#axiomsClean complete_live
#axiomsClean logical_read
#axiomsClean row_read
#axiomsClean read_preserved
#axiomsClean copy_saved_words
#axiomsClean copy_saved_match
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
