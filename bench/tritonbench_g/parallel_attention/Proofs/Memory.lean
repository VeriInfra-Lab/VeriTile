import bench.tritonbench_g.parallel_attention.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

abbrev Memory (α : Type) := Nat → α

def runStores (addr : Nat → Nat) (value : Memory α → Nat → α) : List Nat → Memory α → Memory α
  | [],m => m
  | i::is,m => runStores addr value is (Function.update m (addr i) (value m i))

theorem runStores_frame (addr : Nat → Nat) (value : Memory α → Nat → α)
    (is : List Nat) (m : Memory α) (a : Nat) (outside : ∀ i∈is,addr i≠a) :
    runStores addr value is m a=m a := by
  induction is generalizing m with
  | nil => rfl
  | cons i is ih =>
    simp only [runStores]
    rw [ih _ (by intro j hj; exact outside j (List.mem_cons_of_mem i hj))]
    exact Function.update_of_ne (Ne.symm (outside i (by simp))) _ _

/-- This locality predicate concerns reads from current memory. It is proved
for each source graph; it is not an input/output or successful-run oracle. -/
def ReadsOnly (R : Set Nat) (value : Memory α → Nat → α) : Prop :=
  ∀ m m',Set.EqOn m m' R → ∀ i,value m i=value m' i

theorem runStores_reads (R : Set Nat) (addr : Nat → Nat) (value : Memory α → Nat → α)
    (is : List Nat) (m : Memory α) (disjoint : ∀ i∈is,addr i∉R) :
    Set.EqOn (runStores addr value is m) m R := by
  intro a ha
  apply runStores_frame
  intro i hi eq
  exact disjoint i hi (eq ▸ ha)

theorem runStores_static (R : Set Nat) (addr : Nat → Nat) (value : Memory α → Nat → α)
    (locality : ReadsOnly R value) (is : List Nat) (m initial : Memory α)
    (same : Set.EqOn m initial R) (disjoint : ∀ i∈is,addr i∉R) :
    runStores addr value is m=runStores addr (fun _ => value initial) is m := by
  induction is generalizing m with
  | nil => rfl
  | cons i is ih =>
    simp only [runStores]
    have ev : value m i=value initial i := locality m initial same i
    rw [ev]
    apply ih
    · intro a ha
      rw [Function.update_of_ne (by
        intro eq
        exact disjoint i (by simp) (eq ▸ ha))]
      exact same ha
    · intro j hj
      exact disjoint j (List.mem_cons_of_mem i hj)

theorem static_store_value (addr : Nat → Nat) (value : Nat → α)
    (is : List Nat) (m : Memory α) (target : Nat) (member : target∈is)
    (unique : Function.Injective addr) :
    runStores addr (fun _ => value) is m (addr target)=value target := by
  induction is generalizing m with
  | nil => simp at member
  | cons i is ih =>
    simp only [runStores]
    by_cases later : target∈is
    · exact ih _ later
    · have eq : target=i := by simpa [later] using member
      subst i
      rw [runStores_frame addr (fun _ => value) is _ (addr target) (by
        intro j hj eq
        exact later (unique eq ▸ hj))]
      simp

theorem actual_store_value (R : Set Nat) (addr : Nat → Nat) (value : Memory α → Nat → α)
    (locality : ReadsOnly R value) (is : List Nat) (m : Memory α)
    (disjoint : ∀ i∈is,addr i∉R) (target : Nat) (member : target∈is)
    (unique : Function.Injective addr) :
    runStores addr value is m (addr target)=value m target := by
  rw [runStores_static R addr value locality is m m (by intro a ha; rfl) disjoint]
  exact static_store_value addr (value m) is m target member unique

structure Window where
  base : Nat
  size : Nat

def Window.Cells (w : Window) : Set Nat := {a | w.base≤a ∧ a<w.base+w.size}

theorem window_disjoint (u v : Window) (h : u.base+u.size≤v.base ∨ v.base+v.size≤u.base) :
    Disjoint u.Cells v.Cells := by
  rw [Set.disjoint_left]
  intro a ha hb
  rcases h with h | h <;> simp only [Window.Cells,Set.mem_setOf_eq] at ha hb <;> omega

#axiomsClean runStores_frame
#axiomsClean runStores_reads
#axiomsClean runStores_static
#axiomsClean actual_store_value
#axiomsClean window_disjoint
end
end VeriTile.Bench.Repairs.ParallelAttention
