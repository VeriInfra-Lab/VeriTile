import bench.tritonbench_g.fast_rms_layernorm.Proofs.Storage

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.StorageImage
open VeriTile.Triton Storage
noncomputable section

abbrev Index (h : Call) (p : Phase) := {r : RegionName // r∈buffers p} × Nat × Fin 4
def valid (h : Call) {p : Phase} (i : Index h p) : Prop := i.2.1<cells h i.1.val
def atIndex {h : Call} {p : Phase} (s : Placement h p) (i : Index h p) : Nat :=
  address s i.1.val i.1.property i.2.1 i.2.2

theorem unique (h : Call) (p : Phase) (s : Placement h p) (i j : Index h p)
    (hi : valid h i) (hj : valid h j) (eq : atIndex s i=atIndex s j) : i=j := by
  obtain ⟨r,o,b⟩:=address_unique h p s i.1.val j.1.val i.1.property j.1.property
    i.2.1 j.2.1 hi hj i.2.2 j.2.2 eq
  exact Prod.ext (Subtype.ext r) (Prod.ext o b)

def decode {h : Call} {p : Phase} (s : Placement h p) (addr : Nat) : Option (Index h p) := by
  classical
  exact if ex : ∃ i : Index h p,valid h i ∧ addr=atIndex s i then some ex.choose else none

theorem sound {h : Call} {p : Phase} (s : Placement h p) (addr : Nat) (i : Index h p)
    (eq : decode s addr=some i) : valid h i ∧ addr=atIndex s i := by
  unfold decode at eq
  split at eq
  · rename_i ex
    cases eq
    exact ex.choose_spec
  · cases eq

theorem decode_at (h : Call) (p : Phase) (s : Placement h p) (i : Index h p) (hi : valid h i) :
    decode s (atIndex s i)=some i := by
  unfold decode
  have ex : ∃ j : Index h p,valid h j ∧ atIndex s i=atIndex s j := ⟨i,hi,rfl⟩
  rw [dif_pos ex]
  exact congrArg some (unique h p s ex.choose i ex.choose_spec.1 hi ex.choose_spec.2.symm)

def owns {h : Call} {p : Phase} (s : Placement h p) (addr : Nat) : Prop :=
  ∃ i : Index h p,valid h i ∧ addr=atIndex s i

/-- Encoding the finite source memory on its actual four-byte placements.
Outside the private logical windows, every byte remains the entry byte. -/
def materialize {α β : Type} {h : Call} {p : Phase} (s : Placement h p)
    (encode : α → Fin 4 → β) (entry : Heap β) (values : RegionName → Nat → α) : Heap β :=
  fun addr => match decode s addr with
    | none => entry addr
    | some i => encode (values i.1.val i.2.1) i.2.2

theorem readback {α β : Type} (h : Call) (p : Phase) (s : Placement h p)
    (encode : α → Fin 4 → β) (entry : Heap β) (values : RegionName → Nat → α)
    (r : RegionName) (hr : r∈buffers p) (off : Nat) (inside : off<cells h r) (byte : Fin 4) :
    materialize s encode entry values (address s r hr off byte)=encode (values r off) byte := by
  exact congrArg (fun result => match result with
    | none => entry (address s r hr off byte)
    | some i => encode (values i.1.val i.2.1) i.2.2) (decode_at h p s (⟨r,hr⟩,off,byte) inside)

theorem frame {α β : Type} (h : Call) (p : Phase) (s : Placement h p)
    (encode : α → Fin 4 → β) (entry : Heap β) (values : RegionName → Nat → α)
    (addr : Nat) (outside : ¬owns s addr) : materialize s encode entry values addr=entry addr := by
  unfold owns at outside
  simp only [materialize,decode,dif_neg outside]

def Fresh {h : Call} {p : Phase} (s : Placement h p) (live : Nat → Prop) : Prop :=
  ∀ addr,live addr → ¬owns s addr

theorem caller_frame {α β : Type} (h : Call) (p : Phase) (s : Placement h p)
    (encode : α → Fin 4 → β) (entry : Heap β) (values : RegionName → Nat → α)
    (live : Nat → Prop) (fresh : Fresh s live) (addr : Nat) (active : live addr) :
    materialize s encode entry values addr=entry addr :=
  frame h p s encode entry values addr (fresh addr active)

/-- The source's complete region frame transports to every byte, including
padding, other allocations and all caller backing bytes. -/
theorem source_frame {α β : Type} (h : Call) (p : Phase) (s : Placement h p)
    (encode : α → Fin 4 → β) (entry : Heap β) (before after : RegionName → Nat → α)
    (regionFrame : ∀ r o,¬(r∈outputs p ∧ o<cells h r) → after r o=before r o)
    (addr : Nat) (outside : ¬written s addr) :
    materialize s encode entry after addr=materialize s encode entry before addr := by
  unfold materialize
  cases dec : decode s addr with
  | none => rfl
  | some i =>
    obtain ⟨inside,location⟩:=sound s addr i dec
    change encode (after i.1.val i.2.1) i.2.2=encode (before i.1.val i.2.1) i.2.2
    rw [regionFrame]
    rintro ⟨output,_⟩
    apply outside
    exact ⟨(⟨i.1.val,output⟩,i.2),inside,location⟩

#axiomsClean readback
#axiomsClean caller_frame
#axiomsClean source_frame
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.StorageImage
