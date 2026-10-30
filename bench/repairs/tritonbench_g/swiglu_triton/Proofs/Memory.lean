import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- An owner and the actual byte start of a four-byte FP32 span. Caller
starts may be unaligned; distinct owners may reuse numeric virtual addresses. -/
structure Key where
  owner : Nat
  address : Nat
  deriving DecidableEq

def Key.covers (key : Key) (owner address : Nat) : Prop :=
  owner=key.owner ∧ key.address≤address ∧ address<key.address+4

def Key.disjoint (a b : Key) : Prop :=
  a.owner≠b.owner ∨ a.address+4≤b.address ∨ b.address+4≤a.address

theorem Key.disjoint_symm {a b : Key} (h : a.disjoint b) : b.disjoint a := by
  rcases h with h | h | h
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

theorem Key.disjoint_ne {a b : Key} (h : a.disjoint b) : a≠b := by
  intro same
  subst b
  rcases h with h | h | h <;> simp_all

theorem Key.disjoint_bytes {a b : Key} (h : a.disjoint b) (owner address : Nat)
    (ha : a.covers owner address) : ¬b.covers owner address := by
  rcases ha with ⟨ha,lo,hi⟩
  rintro ⟨hb,lo',hi'⟩
  rcases h with h | h | h
  · exact h (ha.symm.trans hb)
  · omega
  · omega

/-- Ordinary local memory primitives. α is arbitrary (including Real); β is
an observation carrier. Only the FP specialization relates a word to four
finite bytes. There is no encoding of arbitrary Real values into FP32 bytes. -/
structure Ops (M α β : Type) where
  read : M → Key → α
  write : M → Key → α → M
  observe : M → Nat → Nat → β
  read_same : ∀ s key value,read (write s key value) key=value
  read_other : ∀ s writer value observed,observed.disjoint writer →
    read (write s writer value) observed=read s observed
  frame : ∀ s writer value owner address,¬writer.covers owner address →
    observe (write s writer value) owner address=observe s owner address

def stores {M α β ι : Type} (ops : Ops M α β) (key : ι → Key) (values : ι → α) : List ι → M → M
  | [],s => s
  | i::rest,s => stores ops key values rest (ops.write s (key i) (values i))

theorem stores_append {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (first rest : List ι) (s : M) :
    stores ops key values (first++rest) s=stores ops key values rest (stores ops key values first s) := by
  induction first generalizing s with
  | nil => rfl
  | cons i first ih => exact ih _

theorem read_frame {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (observed : Key)
    (outside : ∀ i∈jobs,observed.disjoint (key i)) :
    ops.read (stores ops key values jobs s) observed=ops.read s observed := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,ih _ (fun j hj => outside j (by simp [hj])),ops.read_other _ _ _ _ (outside i (by simp))]

theorem stable_read {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (observed : Key) (expected : α)
    (initial : ops.read s observed=expected)
    (consistent : ∀ i∈jobs,key i=observed → values i=expected)
    (separate : ∀ i∈jobs,key i≠observed → observed.disjoint (key i)) :
    ops.read (stores ops key values jobs s) observed=expected := by
  induction jobs generalizing s with
  | nil => exact initial
  | cons i rest ih =>
    apply ih _ _ (fun j hj => consistent j (by simp [hj])) (fun j hj => separate j (by simp [hj]))
    by_cases same : key i=observed
    · rw [same,ops.read_same,consistent i (by simp) same]
    · rw [ops.read_other _ _ _ _ (separate i (by simp) same)]
      exact initial

theorem readback {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (i : ι) (member : i∈jobs)
    (consistent : ∀ j∈jobs,key j=key i → values j=values i)
    (separate : ∀ j∈jobs,key j≠key i → (key i).disjoint (key j)) :
    ops.read (stores ops key values jobs s) (key i)=values i := by
  obtain ⟨before,after,same⟩ := List.append_of_mem member
  rw [same,stores_append,stores]
  apply stable_read ops key values after _ (key i) (values i) (ops.read_same _ _ _)
  · intro j hj
    exact consistent j (by rw [same]; simp [hj])
  · intro j hj
    exact separate j (by rw [same]; simp [hj])

theorem bytes_frame {M α β ι : Type} (ops : Ops M α β) (key : ι → Key)
    (values : ι → α) (jobs : List ι) (s : M) (owner address : Nat)
    (outside : ∀ i∈jobs,¬(key i).covers owner address) :
    ops.observe (stores ops key values jobs s) owner address=ops.observe s owner address := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [stores,ih _ (fun j hj => outside j (by simp [hj])),ops.frame _ _ _ _ _ (outside i (by simp))]

structure Negation (α : Type) where
  apply : α → α
  involutive : ∀ a,apply (apply a)=a

def Negation.encode {α : Type} (neg : Negation α) (flag : Bool) (a : α) : α :=
  if flag then neg.apply a else a

theorem Negation.encode_twice {α : Type} (neg : Negation α) (flag : Bool) (a : α) :
    neg.encode flag (neg.encode flag a)=a := by cases flag <;> simp [encode,neg.involutive]

structure Allocation where
  object : Nat
  owner : Nat
  baseByte : Nat
  bytes : Nat
  representable : baseByte+bytes≤2^64

def Allocation.words (a : Allocation) : Nat := a.bytes/4

def Allocation.covers (a : Allocation) (owner address : Nat) : Prop :=
  owner=a.owner ∧ a.baseByte≤address ∧ address<a.baseByte+a.bytes

def Allocation.key (a : Allocation) (index : Nat) : Key := ⟨a.owner,a.baseByte+4*index⟩

def Allocation.contains (a : Allocation) (key : Key) : Prop :=
  key.owner=a.owner ∧ a.baseByte≤key.address ∧ key.address+4≤a.baseByte+a.bytes

theorem Allocation.key_contains (a : Allocation) (i : Nat) (inside : i<a.words) :
    a.contains (a.key i) := by simp [contains,key]; unfold words at inside; omega

def Allocation.separate (a b : Allocation) : Prop :=
  a.owner≠b.owner ∨ a.baseByte+a.bytes≤b.baseByte ∨ b.baseByte+b.bytes≤a.baseByte

theorem Allocation.spans_separate (a b : Allocation) (separate : a.separate b)
    (x y : Key) (ax : a.contains x) (by' : b.contains y) : x.disjoint y := by
  rcases ax with ⟨xo,xlow,xhigh⟩
  rcases by' with ⟨yo,ylow,yhigh⟩
  rcases separate with different | before | after
  · exact Or.inl (by intro same; exact different (xo.symm.trans (same.trans yo)))
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

theorem Allocation.keys_separate (a b : Allocation) (separate : a.separate b)
    (x y : Key) (ax : a.contains x) (by' : b.contains y) : x≠y :=
  Key.disjoint_ne (a.spans_separate b separate x y ax by')

theorem Allocation.indices_disjoint (a : Allocation) (i j : Nat) (different : i≠j) :
    (a.key i).disjoint (a.key j) := by
  simp only [Key.disjoint,Allocation.key]
  by_cases h : i<j
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

def offset : List Nat → List Nat → Nat → Nat
  | [],_,_ => 0
  | _::dims,stride::strides,index => index/dims.prod*stride + offset dims strides (index%dims.prod)
  | _::_,[],_ => 0

/-- Full strided tensor metadata. The backing is the complete original
allocation, including storage offset, row gaps and unused trailing bytes.
Negative views change logical reads/writes, never their physical byte spans. -/
structure View where
  object : Nat
  storage : Allocation
  shape : List Nat
  strides : List Nat
  storageOffset : Nat
  negative : Bool
  rank : strides.length=shape.length

def View.numel (v : View) : Nat := v.shape.prod
def View.key (v : View) (i : Nat) : Key :=
  v.storage.key (v.storageOffset+offset v.shape v.strides i)
def View.backed (v : View) : Prop := ∀ i,i<v.numel → v.storage.contains (v.key i)

def logical {M α β : Type} (ops : Ops M α β) (neg : Negation α) (s : M) (v : View) (i : Nat) : α :=
  neg.encode v.negative (ops.read s (v.key i))

def copyback {M α β : Type} (ops : Ops M α β) (neg : Negation α) (v : View)
    (values : Nat → α) (s : M) : M :=
  stores ops v.key (fun i => neg.encode v.negative (values i)) (List.range v.numel) s

theorem copyback_readback {M α β : Type} (ops : Ops M α β) (neg : Negation α) (v : View)
    (values : Nat → α) (s : M) (i : Nat) (active : i<v.numel)
    (separate : ∀ i j,i<v.numel → j<v.numel → i≠j → (v.key i).disjoint (v.key j)) :
    logical ops neg (copyback ops neg v values s) v i=values i := by
  unfold logical copyback
  rw [readback ops v.key _ _ s i (List.mem_range.mpr active)]
  · exact neg.encode_twice _ _
  · intro j hj same
    have equal : j=i := by
      by_contra different
      exact Key.disjoint_ne (separate j i (List.mem_range.mp hj) active different) same
    rw [equal]
  · intro j hj different
    exact separate i j active (List.mem_range.mp hj) (by intro same; subst j; exact different rfl)

#axiomsClean readback
#axiomsClean copyback_readback
#axiomsClean bytes_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Memory
