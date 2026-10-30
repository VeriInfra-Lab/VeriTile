import bench.tritonbench_g.rotary_emb.Proofs.Guards
import bench.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.RotaryEmb.Binding
open Checked
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

abbrev API := VeriTile.Bench.Repairs.SwigluTriton.Memory.Ops
abbrev Job := Work × Bool

def tensorKey (t : Tensor) (offset : Nat) : Key := ⟨t.device,t.base+4*offset⟩
def key (c : Call) (j : Job) := tensorKey (c.tensor (role j.1.key)) (c.shape.cell j.1 j.2)

/-- Mathematical interpretation of the current caller memory, not a runtime
clone or snapshot. Only ordinary aligned four-byte loads appear here. -/
def inputs {M α β : Type} (api : API M α β) (c : Call) (m : M) : Memory α := fun r a =>
  match r with
  | .q => api.read m (tensorKey (c.tensor 0) a)
  | .k => api.read m (tensorKey (c.tensor 1) a)
  | .cos => api.read m (tensorKey (c.tensor 2) a)
  | .sin => api.read m (tensorKey (c.tensor 3) a)
  | .other n => api.read m ⟨n,4*a⟩

theorem inputs_data {M α β : Type} (api : API M α β) (c : Call) (m : M) (which : Bool) (a : Nat) :
    inputs api c m (dataRegion which) a=api.read m (tensorKey (c.tensor (role which)) a) := by
  cases which <;> rfl

def jobs (s : Shape) : List Job := (schedule s) ×ˢ [Bool.false,Bool.true]

theorem member (s : Shape) (j : Job) : j∈jobs s ↔ s.Active j.1 := by
  cases j with
  | mk w odd => cases odd <;> simp [jobs,schedule_mem_iff]

theorem keys_separate (c : Call) (accepted : check c=Bool.true) (a b : Job)
    (ha : a∈jobs c.shape) (hb : b∈jobs c.shape) (different : a≠b) :
    (key c a).disjoint (key c b) := by
  have activeA := (member c.shape a).mp ha
  have activeB := (member c.shape b).mp hb
  by_cases same : a.1.key=b.1.key
  · have unequal : c.shape.cell a.1 a.2≠c.shape.cell b.1 b.2 := by
      intro eq
      have e := cell_injective c.shape (shape_valid c accepted) a.1 b.1 a.2 b.2
        activeA activeB (by rw [same]) eq
      exact different (Prod.ext e.1 e.2)
    simp only [Key.disjoint,key,tensorKey,same]
    by_cases lt : c.shape.cell a.1 a.2<c.shape.cell b.1 b.2
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))
  · have distinct : role a.1.key≠role b.1.key := by
      cases ka : a.1.key <;> cases kb : b.1.key <;> simp_all [role]
    have separate := backing_separate c accepted a.1 activeA _ _ distinct
    have fitA := data_contained c accepted a.1 activeA a.2
    have fitB := data_contained c accepted b.1 activeB b.2
    have baseA := (launch_tensor c accepted a.1 activeA (role a.1.key)).2.1
    have baseB := (launch_tensor c accepted b.1 activeB (role b.1.key)).2.1
    simp only [Checked.separate] at separate
    simp only [Key.disjoint,key,tensorKey]
    rcases separate with first|second
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))

/-- Actual source words are materialized through local stores. The existing
vector/byte closure proves these are the final source stores in any complete
work order, including in-place Q/K execution. No target formula supplies words. -/
def execute {M : Type} (api : API M Native.Word Physical.Byte) (p : Native.Instructions)
    (c : Call) (m : M) : M :=
  let source := Native.selected c.shape p (inputs api c m)
  stores api (key c) (fun j => source (dataRegion j.1.key) (c.shape.cell j.1 j.2)) (jobs c.shape) m

theorem readback {M : Type} (api : API M Native.Word Physical.Byte) (p : Native.Instructions)
    (c : Call) (accepted : check c=Bool.true) (m : M) (w : Work) (active : c.shape.Active w) (odd : Bool) :
    api.read (execute api p c m) (key c (w,odd))=
      Native.pairWord c.shape p w (inputs api c m) odd := by
  unfold execute
  rw [VeriTile.Bench.Repairs.SwigluTriton.Memory.readback api _ _ _ m (w,odd)
    ((member c.shape (w,odd)).mpr active)]
  · exact Native.selected_readback c.shape (shape_valid c accepted) p _ w active odd
  · intro j hj equal
    have same : j=(w,odd) := by
      by_contra ne
      exact Key.disjoint_ne (keys_separate c accepted j (w,odd) hj
        ((member c.shape (w,odd)).mpr active) ne) equal
    rw [same]
  · intro j hj different
    exact keys_separate c accepted (w,odd) j ((member c.shape (w,odd)).mpr active) hj
      (by intro same; subst j; exact different rfl)

def Outside (c : Call) (owner address : Nat) : Prop :=
  ∀ w,c.shape.Active w → ∀ odd,¬(key c (w,odd)).covers owner address

theorem frame {M : Type} (api : API M Native.Word Physical.Byte) (p : Native.Instructions)
    (c : Call) (m : M) (owner address : Nat) (outside : Outside c owner address) :
    api.observe (execute api p c m) owner address=api.observe m owner address := by
  apply bytes_frame
  intro j hj
  exact outside j.1 ((member c.shape j).mp hj) j.2

theorem tail_apart (c : Call) (accepted : check c=Bool.true) (which : Bool) (n h d : Nat)
    (hn : n<c.shape.tokens) (hh : h<c.shape.heads which) (hd : d<c.shape.width)
    (tail : c.shape.rotary≤d) (j : Job) (memberJ : j∈jobs c.shape) :
    (tensorKey (c.tensor (role which)) (c.shape.address which n h d)).disjoint (key c j) := by
  have active := (member c.shape j).mp memberJ
  by_cases same : which=j.1.key
  · have ne : c.shape.address which n h d≠c.shape.cell j.1 j.2 := by
      intro eq
      have outside := tail_outside c.shape (shape_valid c accepted) which n h d hh hd tail j.1 active
      apply outside
      refine ⟨by rw [same],?_⟩
      cases ho : j.2
      · exact Or.inl (by simpa only [ho] using eq)
      · exact Or.inr (by simpa only [ho] using eq)
    simp only [Key.disjoint,key,tensorKey,same] at *
    by_cases lt : c.shape.address j.1.key n h d<c.shape.cell j.1 j.2
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))
  · have distinct : role which≠role j.1.key := by
      cases a : which <;> cases b : j.1.key <;> simp_all [role]
    have separation := backing_separate c accepted j.1 active _ _ distinct
    have fit := data_contained c accepted j.1 active j.2
    have base := (launch_tensor c accepted j.1 active (role j.1.key)).2.1
    have tensorFit := launch_tensor c accepted j.1 active (role which)
    have addressBound : c.shape.address which n h d<c.shape.tokens*c.shape.heads which*c.shape.width := by
      have h1 : n*c.shape.heads which+h<c.shape.tokens*c.shape.heads which := by
        have := Nat.mul_le_mul_right (c.shape.heads which) (by omega : n+1≤c.shape.tokens)
        nlinarith
      have h2 := Nat.mul_le_mul_right c.shape.width (by omega : n*c.shape.heads which+h+1≤c.shape.tokens*c.shape.heads which)
      unfold Shape.address
      nlinarith
    have extentEq : extent c.shape (role which)=c.shape.tokens*c.shape.heads which*c.shape.width := by
      cases which <;> simp [extent,expectedShape,role,Shape.heads,Nat.mul_assoc]
    rw [extentEq] at tensorFit
    simp only [Checked.separate] at separation
    simp only [Key.disjoint,key,tensorKey]
    rcases separation with first|second
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))

theorem tail {M : Type} (api : API M Native.Word Physical.Byte) (p : Native.Instructions)
    (c : Call) (accepted : check c=Bool.true) (m : M) (which : Bool) (n h d : Nat)
    (hn : n<c.shape.tokens) (hh : h<c.shape.heads which) (hd : d<c.shape.width) (ht : c.shape.rotary≤d) :
    api.read (execute api p c m) (tensorKey (c.tensor (role which)) (c.shape.address which n h d))=
      api.read m (tensorKey (c.tensor (role which)) (c.shape.address which n h d)) := by
  apply read_frame
  exact tail_apart c accepted which n h d hn hh hd ht

def expected {M : Type} (api : API M Native.Word Physical.Byte) (p : Native.Instructions)
    (c : Call) (m : M) (which : Bool) (n h d : Nat) : Native.Word :=
  if d<c.shape.rotary then
    Native.pairWord c.shape p ⟨which,n,h,d/2⟩ (inputs api c m) (decide (d%2=1))
  else api.read m (tensorKey (c.tensor (role which)) (c.shape.address which n h d))

theorem all_columns {M : Type} (api : API M Native.Word Physical.Byte) (p : Native.Instructions)
    (c : Call) (accepted : check c=Bool.true) (m : M) (which : Bool) (n h d : Nat)
    (hn : n<c.shape.tokens) (hh : h<c.shape.heads which) (hd : d<c.shape.width) :
    api.read (execute api p c m) (tensorKey (c.tensor (role which)) (c.shape.address which n h d))=
      expected api p c m which n h d := by
  by_cases rotated : d<c.shape.rotary
  · have result := readback api p c accepted m ⟨which,n,h,d/2⟩
      (column_active c.shape (shape_valid c accepted) which n h d hn hh rotated) (decide (d%2=1))
    simp only [key,Shape.cell,column_decomposition] at result
    simpa only [expected,rotated,↓reduceIte] using result
  · simpa [expected,rotated] using tail api p c accepted m which n h d hn hh hd (by omega)

/-- Ordinary byte codec only; it says nothing about the kernel or numerics. -/
structure Codec {M : Type} (api : API M Native.Word Physical.Byte) : Prop where
  bytes : ∀ m k i,api.observe m k.owner (k.address+i.val)=Physical.encode (api.read m k) i

structure Result (M : Type) where
  value : Option Unit
  tensor : Fin 4 → Tensor
  deviceRestored : Nat
  memory : M

def invoke {M : Type} (api : API M Native.Word Physical.Byte) (p : Native.Instructions)
    (c : Call) (callerDevice : Nat) (m : M) : Option (Result M) :=
  if check c then some ⟨none,c.tensor,callerDevice,execute api p c m⟩ else none

theorem complete {M : Type} (api : API M Native.Word Physical.Byte) (codec : Codec api)
    (p : Native.Instructions) (c : Call) (accepted : check c=Bool.true) (callerDevice : Nat) (m : M) :
    ∃ r,invoke api p c callerDevice m=some r ∧ r.value=none ∧ r.tensor=c.tensor ∧
      r.deviceRestored=callerDevice ∧
      (∀ which n h d,n<c.shape.tokens → h<c.shape.heads which → d<c.shape.width →
        api.read r.memory (tensorKey (r.tensor (role which)) (c.shape.address which n h d))=
          expected api p c m which n h d) ∧
      (∀ which n h d i,n<c.shape.tokens → h<c.shape.heads which → d<c.shape.width →
        api.observe r.memory (r.tensor (role which)).device
          ((r.tensor (role which)).base+4*c.shape.address which n h d+i.val)=
          Physical.encode (expected api p c m which n h d) i) ∧
      (∀ owner address,Outside c owner address → api.observe r.memory owner address=api.observe m owner address) := by
  refine ⟨⟨none,c.tensor,callerDevice,execute api p c m⟩,by simp [invoke,accepted],rfl,rfl,rfl,?_,?_,?_⟩
  · exact all_columns api p c accepted m
  · intro which n h d i hn hh hd
    change api.observe (execute api p c m)
      (tensorKey (c.tensor (role which)) (c.shape.address which n h d)).owner
      ((tensorKey (c.tensor (role which)) (c.shape.address which n h d)).address+i.val)=_
    rw [codec.bytes,all_columns api p c accepted m which n h d hn hh hd]
  · exact frame api p c m

#axiomsClean keys_separate
#axiomsClean all_columns
#axiomsClean complete
#specNonCircular expected avoiding [execute,invoke,Native.selected,source]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb.Binding
