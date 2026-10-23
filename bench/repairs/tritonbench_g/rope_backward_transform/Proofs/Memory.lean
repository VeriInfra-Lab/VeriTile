import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Grid
import bench.repairs.tritonbench_g.rope_embedding.Proofs.Memory
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Memory
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API Write runWrites writes_read_frame writes_byte_frame writes_readback)
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Location where
  key : Bool
  row : Nat
  side : Bool
  (head lane : Nat)
  deriving DecidableEq

def Location.column (r : Request) (l : Location) := l.lane+if l.side then r.half else 0
def Active (r : Request) (l : Location) := l.head<r.heads l.key ∧ l.lane<r.half
instance (r : Request) (l : Location) : Decidable (Active r l) := inferInstanceAs (Decidable (_∧_))
def locations (r : Request) (rows : List Nat) : List Location :=
  rows.reverse.flatMap fun a => [Bool.false,Bool.true].flatMap fun k =>
    [Bool.false,Bool.true].flatMap fun side => (List.range (r.paddedHeads k)).flatMap fun h =>
      (List.range (r.paddedDim/2)).map fun j => ⟨k,a,side,h,j⟩

theorem mem_locations (r : Request) (xs : List Nat) (l : Location) :
    l∈locations r xs ↔ l.row∈xs ∧ l.head<r.paddedHeads l.key ∧ l.lane<r.paddedDim/2 := by
  cases l with | mk k a side h j =>
    cases k <;> cases side <;> simp [locations]

theorem column_bound (r : Request) (l : Location) (ha : Active r l) : l.column r<r.dimension :=
  half_column r l.side l.lane ha.2

theorem coverage (r : Request) (xs : List Nat) (k : Bool) (a h c : Nat)
    (ha : a∈xs) (hh : h<r.heads k) (hc : c<2*r.half) :
    ∃ l∈locations r xs,Active r l ∧ l.key=k ∧ l.row=a ∧ l.head=h ∧ l.column r=c := by
  let j := if c<r.half then c else c-r.half
  have hj : j<r.half := by dsimp [j]; split <;> omega
  let l : Location := ⟨k,a,decide (r.half≤c),h,j⟩
  refine ⟨l,(mem_locations r xs l).mpr ⟨ha,lt_of_lt_of_le hh (pad_covers _),
    lt_of_lt_of_le hj (half_covers r)⟩,⟨hh,hj⟩,rfl,rfl,rfl,?_⟩
  dsimp [Location.column,l,j]
  split <;> simp_all <;> omega

def Separated (r : Request) (base : Bool→Nat) :=
  base Bool.false+4*r.size Bool.false≤base Bool.true ∨
    base Bool.true+4*r.size Bool.true≤base Bool.false

theorem keys_apart (r : Request) (base : Bool→Nat) (sep : Separated r base)
    (k k' : Bool) (a a' : Nat) (hk : k≠k') (ha : a<r.size k) (ha' : a'<r.size k') :
    base k+4*a+4≤base k'+4*a' ∨ base k'+4*a'+4≤base k+4*a := by
  cases k <;> cases k' <;> simp_all [Separated] <;> omega

def events {α : Type} [Inhabited α] (project : α→β) (op : Operations α) (r : Request)
    (m : Source.Memory α) (base : Bool→Nat) (ls : List Location) : List (Write β) :=
  ls.filterMap fun l => if Active r l then some
    ⟨base l.key+4*r.offset l.key l.row l.head (l.column r),
      project (scalarResult op r m l.key l.row l.head (l.column r))⟩ else none

def run {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (m : Source.Memory α) (base : Bool→Nat) (xs : List Nat) (store : Store) :=
  runWrites api (events project op r m base (locations r xs)) store

theorem mem_events {α : Type} [Inhabited α] (project : α→β) (op : Operations α) (r : Request)
    (m : Source.Memory α) (base : Bool→Nat) (ls : List Location) (e : Write β) :
    e∈events project op r m base ls ↔ ∃ l∈ls,Active r l ∧ e=
      ⟨base l.key+4*r.offset l.key l.row l.head (l.column r),
        project (scalarResult op r m l.key l.row l.head (l.column r))⟩ := by
  simp only [events,List.mem_filterMap]
  constructor
  · rintro ⟨l,hl,he⟩
    split at he
    · exact ⟨l,hl,‹_›,(Option.some.inj he).symm⟩
    · simp at he
  · rintro ⟨l,hl,ha,rfl⟩
    exact ⟨l,hl,by simp [ha]⟩

theorem readback {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (m : Source.Memory α) (base : Bool→Nat) (xs : List Nat) (store : Store)
    (sep : Separated r base) (valid : ∀ row∈xs,row<r.rows)
    (k : Bool) (a h c : Nat) (ha : a∈xs) (hh : h<r.heads k) (hc : c<2*r.half) :
    api.read (run api project op r m base xs store) (base k+4*r.offset k a h c)=
      project (scalarResult op r m k a h c) := by
  apply writes_readback
  · obtain ⟨l,hl,hal,hk,hr,hh',hc'⟩ := coverage r xs k a h c ha hh hc
    refine ⟨_,(mem_events ..).mpr ⟨l,hl,hal,rfl⟩,?_,?_⟩ <;> simp only [hk,hr,hh',hc']
  · intro e he
    obtain ⟨l,hl,hal,rfl⟩ := (mem_events ..).mp he
    have hr := valid l.row ((mem_locations r xs l).mp hl).1
    have hlc := column_bound r l hal
    have hcD : c<r.dimension := by
      have : 2*r.half≤r.dimension := by unfold Request.half; omega
      omega
    by_cases hk : l.key=k
    · subst k
      by_cases eq : r.offset l.key l.row l.head (l.column r)=r.offset l.key a h c
      · obtain ⟨hr',hh',hc'⟩ := offset_injective r l.key hal.1 hlc hh hcD eq
        left
        simp [hr',hh',hc']
      · right
        dsimp only
        omega
    · right
      exact keys_apart r base sep l.key k _ _ hk
        (offset_bound r l.key l.row l.head (l.column r) hr hal.1 hlc)
        (offset_bound r k a h c (valid a ha) hh hcD)

theorem coordinate_frame {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (base : Bool→Nat)
    (xs : List Nat) (store : Store) (sep : Separated r base) (valid : ∀ row∈xs,row<r.rows)
    (k : Bool) (a h c : Nat) (ha : a<r.rows) (hh : h<r.heads k) (hc : c<r.dimension)
    (other : a∉xs ∨ 2*r.half≤c) :
    api.read (run api project op r m base xs store) (base k+4*r.offset k a h c)=
      api.read store (base k+4*r.offset k a h c) := by
  apply writes_read_frame
  intro e he
  obtain ⟨l,hl,hal,rfl⟩ := (mem_events ..).mp he
  have hr := valid l.row ((mem_locations r xs l).mp hl).1
  have hlc := column_bound r l hal
  by_cases hk : l.key=k
  · subst k
    have ne : r.offset l.key l.row l.head (l.column r)≠r.offset l.key a h c := by
      intro eq
      obtain ⟨hr',hh',hc'⟩ := offset_injective r l.key hal.1 hlc hh hc eq
      rcases other with hn | ht
      · exact hn (hr' ▸ ((mem_locations r xs l).mp hl).1)
      · have hj := hal.2
        unfold Location.column at hc'
        split at hc' <;> omega
    dsimp only
    omega
  · exact keys_apart r base sep l.key k _ _ hk
      (offset_bound r l.key l.row l.head (l.column r) hr hal.1 hlc)
      (offset_bound r k a h c ha hh hc)

/-- Exact caller-byte complement of the active writes, including odd tails. -/
def Outside (r : Request) (base : Bool→Nat) (xs : List Nat) (b : Nat) :=
  ∀ l∈locations r xs,Active r l →
    b<base l.key+4*r.offset l.key l.row l.head (l.column r) ∨
      base l.key+4*r.offset l.key l.row l.head (l.column r)+4≤b

theorem byte_frame {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (m : Source.Memory α) (base : Bool→Nat) (xs : List Nat) (store : Store)
    (b : Nat) (hb : Outside r base xs b) :
    api.byte (run api project op r m base xs store) b=api.byte store b := by
  apply writes_byte_frame
  intro e he
  obtain ⟨l,hl,hal,rfl⟩ := (mem_events ..).mp he
  exact hb l hl hal

theorem read_frame {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (m : Source.Memory α) (base : Bool→Nat) (xs : List Nat) (store : Store)
    (a : Nat) (ha : ∀ k, a+4≤base k ∨ base k+4*r.size k≤a) (valid : ∀ row∈xs,row<r.rows) :
    api.read (run api project op r m base xs store) a=api.read store a := by
  apply writes_read_frame
  intro e he
  obtain ⟨l,hl,hal,rfl⟩ := (mem_events ..).mp he
  have ho := offset_bound r l.key l.row l.head (l.column r)
    (valid l.row ((mem_locations r xs l).mp hl).1) hal.1 (column_bound r l hal)
  have hk := ha l.key
  dsimp only
  omega

theorem typed_bridge {α : Type} [Inhabited α] (api : API β Store Byte) (project : α→β) (op : Operations α)
    (r : Request) (m : Source.Memory α) (base : Bool→Nat) (xs : List Nat) (store : Store)
    (sep : Separated r base) (valid : ∀ row∈xs,row<r.rows) (nd : xs.Nodup)
    (k : Bool) (a h c : Nat) (ha : a∈xs) (hh : h<r.heads k) (hc : c<2*r.half) :
    api.read (run api project op r m base xs store) (base k+4*r.offset k a h c)=
      project ((grid op.backend r m xs (region k) (r.offset k a h c)).read .fp32) := by
  rw [readback api project op r m base xs store sep valid k a h c ha hh hc,
    grid_readback op r m xs nd a h c k ha hh]
  have : 2*r.half≤r.dimension := by unfold Request.half; omega
  omega

theorem no_launch_events {α : Type} [Inhabited α] (project : α→β) (op : Operations α)
    (r : Request) (m : Source.Memory α) (base : Bool→Nat) (xs : List Nat)
    (h : xs=[] ∨ r.half=0 ∨ (r.qHeads=0 ∧ r.kHeads=0)) : events project op r m base (locations r xs)=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro e he
  obtain ⟨l,hl,hal,eq⟩ := (mem_events ..).mp he
  rcases h with hx | hh | hk
  · subst xs; simp [locations] at hl
  · have := hal.2; omega
  · have := hal.1; cases l.key <;> simp_all [Request.heads]

#axiomsClean readback
#axiomsClean coordinate_frame
#axiomsClean byte_frame
#axiomsClean typed_bridge
#axiomsClean no_launch_events
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.Memory
