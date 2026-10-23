import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.NativeGrid

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Storage
open VeriTile.Triton Native
noncomputable section

inductive Phase | forward | backward deriving DecidableEq
def buffers : Phase → List RegionName
  | .forward => ["x","w","y","r"]
  | .backward => ["x","w","r","dx","dw"]
def outputs : Phase → List RegionName
  | .forward => ["y","r"]
  | .backward => ["dx","dw"]
def cells (h : Call) (r : RegionName) : Nat :=
  if r="w" ∨ r="dw" then h.cols else if r="r" then h.rows else h.cells

theorem output_member (p : Phase) (r : RegionName) (hr : r∈outputs p) : r∈buffers p := by
  cases p <;> simp_all [outputs,buffers]

/-- Only roles allocated in the current phase carry allocation metadata.
Backward's x/w/r entries will be tied to the actual forward saved package. -/
structure Placement (h : Call) (p : Phase) where
  allocation : (r : RegionName) → r∈buffers p → Allocation (cells h r)
  identities : ∀ r hr s hs,r≠s → (allocation r hr).storageId≠(allocation s hs).storageId
  separate : ∀ r hr s hs,r≠s →
    (allocation r hr).buffer.base+(allocation r hr).buffer.extent*4≤(allocation s hs).buffer.base ∨
    (allocation s hs).buffer.base+(allocation s hs).buffer.extent*4≤(allocation r hr).buffer.base

def address {h : Call} {p : Phase} (storage : Placement h p)
    (r : RegionName) (hr : r∈buffers p) (off : Nat) (byte : Fin 4) : Nat :=
  (storage.allocation r hr).buffer.base+off*4+byte.val

theorem address_unique (h : Call) (p : Phase) (storage : Placement h p)
    (r s : RegionName) (hr : r∈buffers p) (hs : s∈buffers p)
    (o k : Nat) (ho : o<cells h r) (hk : k<cells h s) (a b : Fin 4)
    (eq : address storage r hr o a=address storage s hs k b) : r=s ∧ o=k ∧ a=b := by
  have ro := ho.trans_le (storage.allocation r hr).holds
  have sk := hk.trans_le (storage.allocation s hs).holds
  have aa:=a.isLt
  have bb:=b.isLt
  have regions : r=s := by
    by_contra ne
    rcases storage.separate r hr s hs ne with sep|sep <;> unfold address at eq <;> omega
  subst s
  refine ⟨rfl,?_,?_⟩
  · unfold address at eq
    omega
  · apply Fin.ext
    unfold address at eq
    omega

abbrev Heap (β : Type) := Nat → β
abbrev Index (h : Call) (p : Phase) := {r : RegionName // r∈outputs p} × Nat × Fin 4

def valid (h : Call) {p : Phase} (i : Index h p) : Prop := i.2.1<cells h i.1.val
def atIndex {h : Call} {p : Phase} (storage : Placement h p) (i : Index h p) : Nat :=
  address storage i.1.val (output_member p i.1.val i.1.property) i.2.1 i.2.2

theorem index_unique (h : Call) (p : Phase) (storage : Placement h p)
    (i j : Index h p) (hi : valid h i) (hj : valid h j) (eq : atIndex storage i=atIndex storage j) : i=j := by
  obtain ⟨r,o,b⟩ := address_unique h p storage i.1.val j.1.val
    (output_member p _ i.1.property) (output_member p _ j.1.property) i.2.1 j.2.1 hi hj i.2.2 j.2.2 eq
  exact Prod.ext (Subtype.ext r) (Prod.ext o b)

def decodeIndex {h : Call} {p : Phase} (storage : Placement h p) (addr : Nat) : Option (Index h p) := by
  classical
  exact if ex : ∃ i : Index h p,valid h i ∧ addr=atIndex storage i then some ex.choose else none

theorem decode_some (h : Call) (p : Phase) (storage : Placement h p) (addr : Nat) (i : Index h p)
    (eq : decodeIndex storage addr=some i) : valid h i ∧ addr=atIndex storage i := by
  unfold decodeIndex at eq
  split at eq
  · rename_i ex
    cases eq
    exact ex.choose_spec
  · cases eq

theorem decode_at (h : Call) (p : Phase) (storage : Placement h p) (i : Index h p) (hi : valid h i) :
    decodeIndex storage (atIndex storage i)=some i := by
  unfold decodeIndex
  have ex : ∃ j : Index h p,valid h j ∧ atIndex storage i=atIndex storage j := ⟨i,hi,rfl⟩
  rw [dif_pos ex]
  exact congrArg some (index_unique h p storage ex.choose i ex.choose_spec.1 hi ex.choose_spec.2.symm)

def commit {β γ : Type} {h : Call} {p : Phase} (storage : Placement h p)
    (encode : γ → Fin 4 → β) (before : Heap β) (after : RegionName → Nat → γ) : Heap β :=
  fun addr => match decodeIndex storage addr with
    | none => before addr
    | some i => encode (after i.1.val i.2.1) i.2.2

theorem commit_output {β γ : Type} (h : Call) (p : Phase) (storage : Placement h p)
    (encode : γ → Fin 4 → β) (before : Heap β) (after : RegionName → Nat → γ)
    (r : RegionName) (hr : r∈outputs p) (o : Nat) (ho : o<cells h r) (byte : Fin 4) :
    commit storage encode before after (address storage r (output_member p r hr) o byte)=encode (after r o) byte := by
  exact congrArg (fun result => match result with
    | none => before (address storage r (output_member p r hr) o byte)
    | some i => encode (after i.1.val i.2.1) i.2.2) (decode_at h p storage (⟨r,hr⟩,o,byte) ho)

def written {h : Call} {p : Phase} (storage : Placement h p) (addr : Nat) : Prop :=
  ∃ i : Index h p,valid h i ∧ addr=atIndex storage i

theorem commit_frame {β γ : Type} (h : Call) (p : Phase) (storage : Placement h p)
    (encode : γ → Fin 4 → β) (before : Heap β) (after : RegionName → Nat → γ)
    (addr : Nat) (outside : ¬written storage addr) : commit storage encode before after addr=before addr := by
  unfold written at outside
  simp only [commit,decodeIndex,dif_neg outside]

#axiomsClean address_unique
#axiomsClean commit_output
#axiomsClean commit_frame
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Storage
