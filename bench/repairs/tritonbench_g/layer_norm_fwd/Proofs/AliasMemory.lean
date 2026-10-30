import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! A view maps its own logical offsets to physical storage. Read views may
have different bases and overlap arbitrarily. Only a written cell must have a
unique physical preimage among live cells, the ordinary fresh-output contract.
This is a finite typed-cell physical model indexed by byte-addressed cell
starts. It does not claim an executable raw-byte codec implementation. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.AliasMemory
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

abbrev Key := RegionName × Nat
structure Layout where
  capacity : Nat
  address : Key → Nat
  live : Key → Prop
  bounded : ∀ k, live k → address k < capacity

/-- The library's tensor/view byte address, with independent view bases. -/
def viewAddress (base width : RegionName → Nat) (k : Key) : Nat :=
  base k.1 + width k.1*k.2

/-- Fresh output cells are unique against all live views. There is deliberately
no injectivity requirement between two read-only cells. -/
def FreshWrites (a : Layout) (writes : Key → Prop) : Prop :=
  ∀ w, writes w → a.live w ∧
    ∀ k, a.live k → a.address k = a.address w → k = w

abbrev Physical (a : Layout) := Fin a.capacity → MemCell

def read (a : Layout) (p : Physical a) (k : Key) : MemCell :=
  if h : a.live k then p ⟨a.address k, a.bounded k h⟩ else MemCell.real 0

def pull (a : Layout) (p : Physical a) (template : BlockState) : BlockState :=
  { template with mem := fun r o => read a p (r,o) }

/-- A physical read shared by overlapping views has one value. -/
theorem alias_read (a : Layout) (p : Physical a) (k l : Key)
    (hk : a.live k) (hl : a.live l) (h : a.address k = a.address l) :
    read a p k = read a p l := by
  simp only [read, dif_pos hk, dif_pos hl]
  congr 1
  exact Fin.ext h

def write (a : Layout) (p : Physical a) (k : Key) (v : MemCell) : Physical a :=
  fun i => if i.val = a.address k then v else p i

/-- One actual store commutes with physical realization, including all
aliased live reads and the full complement. No read/read separation is used. -/
theorem store_commutes (a : Layout) (p : Physical a) (s : BlockState)
    (writes : Key → Prop) (hf : FreshWrites a writes)
    (k : Key) (hk : writes k) (v : MemCell) :
    pull a (write a p k v) s = Memory.writeCell (pull a p s) k.1 k.2 v := by
  apply BlockState.ext
  · intro r o
    change read a (write a p k v) (r,o) =
      if r=k.1 ∧ o=k.2 then v else read a p (r,o)
    by_cases he : (r,o) = k
    · obtain ⟨rfl,rfl⟩ := Prod.mk.inj he
      simp [read,write,(hf k hk).1]
    · have hp : ¬(r=k.1 ∧ o=k.2) := by intro hc; exact he (Prod.ext hc.1 hc.2)
      rw [if_neg hp]
      by_cases hl : a.live (r,o)
      · have ha : a.address (r,o) ≠ a.address k := fun h => he ((hf k hk).2 _ hl h)
        simp [read,write,hl,ha]
      · simp [read,hl]
  · intros; rfl
  · intros; rfl
  · intros; rfl
  · intros; rfl

structure Event where
  key : Key
  value : MemCell

def runPhysical (a : Layout) (p : Physical a) (events : List Event) : Physical a :=
  events.foldl (fun memory event => write a memory event.key event.value) p

def runTagged (s : BlockState) (events : List Event) : BlockState :=
  events.foldl (fun state event => Memory.writeCell state event.key.1 event.key.2 event.value) s

/-- Complete ordered store-trace simulation, not a final-output assumption. -/
theorem trace_commutes (a : Layout) (p : Physical a) (s : BlockState)
    (writes : Key → Prop) (hf : FreshWrites a writes) (events : List Event)
    (he : ∀ e ∈ events, writes e.key) :
    pull a (runPhysical a p events) s = runTagged (pull a p s) events := by
  induction events generalizing p s with
  | nil => rfl
  | cons e es ih =>
    simp only [runPhysical,runTagged,List.foldl_cons]
    change pull a (runPhysical a (write a p e.key e.value) es) s = _
    rw [ih _ s (fun e' h => he e' (List.mem_cons_of_mem _ h))]
    rw [store_commutes a p s writes hf e.key (he e (List.mem_cons_self ..)) e.value]
    rfl

/-- Physical frame for every cell outside the actual emitted store addresses. -/
theorem trace_frame (a : Layout) (p : Physical a) (events : List Event)
    (i : Fin a.capacity) (h : ∀ e ∈ events, i.val ≠ a.address e.key) :
    runPhysical a p events i = p i := by
  induction events generalizing p with
  | nil => rfl
  | cons e es ih =>
    simp only [runPhysical,List.foldl_cons]
    change runPhysical a (write a p e.key e.value) es i = p i
    rw [ih _ (fun e' h' => h e' (List.mem_cons_of_mem _ h'))]
    exact if_neg (h e (List.mem_cons_self ..))

/-- Full-grid write footprint already established from the source. -/
def Writes (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (k : Key) : Prop :=
  ∃ row, row<M ∧ Contract.written f r stride M N row k.1 k.2

/-- Finite physical final memory, writing only the actual source write set.
Uniqueness is proved from FreshWrites; reads are never copied or separated. -/
def realise (a : Layout) (initial : Physical a) (final : BlockState)
    (writes : Key → Prop) : Physical a := fun i =>
  if h : ∃ k, writes k ∧ a.address k = i.val then
    final.mem (Classical.choose h).1 (Classical.choose h).2
  else initial i

theorem realise_written (a : Layout) (initial : Physical a) (final : BlockState)
    (writes : Key → Prop) (hf : FreshWrites a writes) (k : Key) (hk : writes k) :
    read a (realise a initial final writes) k = final.mem k.1 k.2 := by
  simp only [read,dif_pos (hf k hk).1,realise]
  have hex : ∃ l, writes l ∧ a.address l = a.address k := ⟨k,hk,rfl⟩
  rw [dif_pos hex]
  have chosen := Classical.choose_spec hex
  have he : Classical.choose hex = k :=
    (hf k hk).2 _ (hf _ chosen.1).1 chosen.2
  rw [he]

theorem realise_unwritten (a : Layout) (initial : Physical a) (final : BlockState)
    (writes : Key → Prop) (i : Fin a.capacity)
    (h : ∀ k, writes k → a.address k ≠ i.val) :
    realise a initial final writes i = initial i := by
  apply dif_neg
  rintro ⟨k,hk,ha⟩
  exact h k hk ha

/-- Exact pullback of every live view after a source run with its full frame.
This is the alias-aware physical refinement missing from the disjoint FlatAlloc
bridge. It requires only fresh output writes, not disjoint input allocations. -/
theorem realise_pull (a : Layout) (initial : Physical a) (s final : BlockState)
    (writes : Key → Prop) (hf : FreshWrites a writes)
    (hframe : ∀ k, ¬writes k → final.mem k.1 k.2 = (pull a initial s).mem k.1 k.2)
    (k : Key) (hl : a.live k) :
    read a (realise a initial final writes) k = final.mem k.1 k.2 := by
  by_cases hw : writes k
  · exact realise_written a initial final writes hf k hw
  · rw [hframe k hw]
    change read a (realise a initial final writes) k = read a initial k
    simp only [read,dif_pos hl]
    apply realise_unwritten
    intro w hw' ha
    have he : k=w := (hf w hw').2 k hl ha.symm
    exact hw (he.symm ▸ hw')

/-- Whole state realization, including out-of-view cells and all registers. -/
theorem realise_pull_state (a : Layout) (initial : Physical a) (s final : BlockState)
    (writes : Key → Prop) (hf : FreshWrites a writes)
    (hframe : ∀ k, ¬writes k → final.mem k.1 k.2 = (pull a initial s).mem k.1 k.2) :
    pull a (realise a initial final writes) final = final := by
  apply BlockState.ext
  · intro r o
    change read a (realise a initial final writes) (r,o) = final.mem r o
    by_cases hl : a.live (r,o)
    · exact realise_pull a initial s final writes hf hframe (r,o) hl
    · have hw : ¬writes (r,o) := fun h => hl (hf _ h).1
      rw [hframe (r,o) hw]
      simp [pull,read,hl]
  · intros; rfl
  · intros; rfl
  · intros; rfl
  · intros; rfl

/-- All actual source reads are preserved, even when their base-offset views
partially overlap each other. This also preserves the original input alias. -/
theorem realise_read_preserved (a : Layout) (initial : Physical a) (final : BlockState)
    (writes : Key → Prop) (hf : FreshWrites a writes)
    (k : Key) (hl : a.live k) (hr : ¬writes k) :
    read a (realise a initial final writes) k = read a initial k := by
  simp only [read,dif_pos hl]
  apply realise_unwritten
  intro w hw ha
  have he := (hf w hw).2 k hl ha.symm
  exact hr (he.symm ▸ hw)

/-- Concrete overlapping inputs with different bases satisfy the memory model.
Output address 8 is fresh, while input views at bases 0 and 1 overlap. -/
def overlapExample : Layout where
  capacity := 16
  address := fun k => if k.1="out" then 8+k.2 else if k.1="x1" then 1+k.2 else k.2
  live := fun k => (k.1="x" ∨ k.1="x1" ∨ k.1="out") ∧ k.2<3
  bounded := by intro k hk; change (if k.1="out" then 8+k.2 else if k.1="x1" then 1+k.2 else k.2)<16; split_ifs <;> omega

theorem overlap_addresses :
    overlapExample.address ("x",1) = overlapExample.address ("x1",0) := by decide

theorem overlap_fresh : FreshWrites overlapExample (fun k => k.1="out" ∧ k.2<3) := by
  intro w hw
  refine ⟨by simp [overlapExample,hw.1,hw.2],?_⟩
  intro k hk he
  rcases hk.1 with hx | hx | hx
  · simp [overlapExample,hw.1,hx] at he
    have h1 := hk.2; have h2 := hw.2
    omega
  · simp [overlapExample,hw.1,hx] at he
    have h1 := hk.2; have h2 := hw.2
    omega
  · apply Prod.ext
    · exact hx.trans hw.1.symm
    · simpa [overlapExample,hw.1,hx] using he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.AliasMemory
