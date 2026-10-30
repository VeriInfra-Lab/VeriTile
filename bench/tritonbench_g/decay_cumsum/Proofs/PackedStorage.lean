import bench.tritonbench_g.decay_cumsum.Proofs.Roles
import bench.tritonbench_g.decay_cumsum.Proofs.Grid
import bench.tritonbench_g.decay_cumsum.Proofs.NativeIndex
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.Storage

namespace VeriTile.Bench.Repairs.DecayCumsum.PackedStorage
open VeriTile.Triton Geometry Grid
noncomputable section
set_option maxHeartbeats 5000000

namespace Shared
abbrev address := VeriTile.Bench.Repairs.ChunkCumsumVector.Storage.address
abbrev address_injective := VeriTile.Bench.Repairs.ChunkCumsumVector.Storage.address_injective
end Shared

structure Placement (h : Call) (kind : Kind) where
  allocation : RegionName → NativeIndex.Allocation h
  extent : ∀ r∈Roles.buffers kind, (allocation r).buffer.extent=h.cells
  separate : ∀ r∈Roles.buffers kind, ∀ s∈Roles.buffers kind, r≠s →
    (allocation r).buffer.base+(allocation r).buffer.extent*4≤(allocation s).buffer.base ∨
    (allocation s).buffer.base+(allocation s).buffer.extent*4≤(allocation r).buffer.base

def address {h : Call} {kind : Kind} (storage : Placement h kind) (r : RegionName) (off : Nat) : Nat :=
  Shared.address (storage.allocation r).buffer off

theorem address_unique (h : Call) (kind : Kind) (storage : Placement h kind)
    (r s : RegionName) (hr : r∈Roles.buffers kind) (hs : s∈Roles.buffers kind)
    (o k : Nat) (ho : o<h.cells) (hk : k<h.cells) (eq : address storage r o=address storage s k) :
    r=s ∧ o=k := by
  have regions : r=s := by
    by_contra hn
    have er := storage.extent r hr
    have es := storage.extent s hs
    have br := (storage.allocation r).fp32Bytes
    have bs := (storage.allocation s).fp32Bytes
    unfold address Shared.address VeriTile.Bench.Repairs.ChunkCumsumVector.Storage.address at eq
    rw [br,bs] at eq
    rcases storage.separate r hr s hs hn with disjoint|disjoint <;> simp only [er,es] at disjoint <;> omega
  subst s
  exact ⟨rfl,Shared.address_injective (storage.allocation r).buffer eq⟩

def shift (h : Call) (p : Phase) : Nat := p.start*h.width

def relative (h : Call) (p : Phase) (prog : Program h p) (t j : Nat) : Nat :=
  prog.row*h.headStride+prog.chunk*p.span*h.width+prog.feature*h.BK+j+t*h.width

theorem relocated (h : Call) (p : Phase) (prog : Program h p) (t j : Nat) :
    shift h p+relative h p prog t j=Geometry.offset h p prog t j := by
  unfold shift relative Geometry.offset Geometry.time Geometry.column
  ring

abbrev Heap (β : Type) := Nat → β

def lift {β : Type} {h : Call} {kind : Kind} (storage : Placement h kind) (p : Phase) (heap : Heap β) : RegionName → Nat → β :=
  fun r o => heap (address storage r (shift h p+o))

def decode {h : Call} {kind : Kind} (storage : Placement h kind) (p : Phase) (addr : Nat) : Option (RegionName × Nat) := by
  classical
  exact if ex : ∃ pair : RegionName × Nat, pair.1∈Roles.outputs kind ∧ shift h p+pair.2<h.cells ∧
      addr=address storage pair.1 (shift h p+pair.2) then some ex.choose else none

theorem decode_some (h : Call) (kind : Kind) (storage : Placement h kind) (p : Phase)
    (addr : Nat) (pair : RegionName × Nat) (he : decode storage p addr=some pair) :
    pair.1∈Roles.outputs kind ∧ shift h p+pair.2<h.cells ∧ addr=address storage pair.1 (shift h p+pair.2) := by
  unfold decode at he
  split at he
  · rename_i ex
    cases he
    exact ex.choose_spec
  · cases he

theorem decode_address (h : Call) (kind : Kind) (storage : Placement h kind) (p : Phase)
    (r : RegionName) (hr : r∈Roles.outputs kind) (o : Nat) (ho : shift h p+o<h.cells) :
    decode storage p (address storage r (shift h p+o))=some (r,o) := by
  unfold decode
  have ex : ∃ pair : RegionName × Nat, pair.1∈Roles.outputs kind ∧ shift h p+pair.2<h.cells ∧
      address storage r (shift h p+o)=address storage pair.1 (shift h p+pair.2) := ⟨(r,o),hr,ho,rfl⟩
  rw [dif_pos ex]
  obtain ⟨same,offsets⟩ := address_unique h kind storage r ex.choose.1 (Roles.output_member kind r hr)
    (Roles.output_member kind ex.choose.1 ex.choose_spec.1) _ _ ho ex.choose_spec.2.1 ex.choose_spec.2.2
  apply congrArg some
  exact Prod.ext same.symm (by omega)

def commit {β : Type} {h : Call} {kind : Kind} (storage : Placement h kind) (p : Phase)
    (before : Heap β) (after : RegionName → Nat → β) : Heap β :=
  fun addr => match decode storage p addr with
    | none => before addr
    | some pair => after pair.1 pair.2

theorem commit_output {β : Type} (h : Call) (kind : Kind) (storage : Placement h kind) (p : Phase)
    (before : Heap β) (after : RegionName → Nat → β) (r : RegionName) (hr : r∈Roles.outputs kind)
    (o : Nat) (ho : shift h p+o<h.cells) :
    commit storage p before after (address storage r (shift h p+o))=after r o := by
  simp only [commit,decode_address h kind storage p r hr o ho]

def footprint (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h) (addr : Nat) : Prop :=
  ∃ r∈Roles.buffers kind, ∃ t : Fin w.1.span, ∃ j : Fin h.BK,
    addr=address storage r (Geometry.offset h w.1 w.2 t.val j.val)

def writes (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h) (addr : Nat) : Prop :=
  ∃ r∈Roles.outputs kind, ∃ t : Fin w.1.span, ∃ j : Fin h.BK,
    addr=address storage r (Geometry.offset h w.1 w.2 t.val j.val)

theorem writes_footprint (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h) (addr : Nat)
    (hw : writes h kind storage w addr) : footprint h kind storage w addr := by
  obtain ⟨r,hr,t,j,eq⟩ := hw
  exact ⟨r,Roles.output_member kind r hr,t,j,eq⟩

def TileFrame {β : Type} (h : Call) (kind : Kind) (w : Work h)
    (before after : RegionName → Nat → β) : Prop :=
  ∀ r o, ¬(r∈Roles.outputs kind ∧ ∃ t : Fin w.1.span, ∃ j : Fin h.BK,
    o=relative h w.1 w.2 t.val j.val) → after r o=before r o

theorem commit_frame {β : Type} (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h)
    (before : Heap β) (after : RegionName → Nat → β)
    (frame : TileFrame h kind w (lift storage w.1 before) after) (addr : Nat)
    (outside : ¬writes h kind storage w addr) : commit storage w.1 before after addr=before addr := by
  unfold commit
  cases he : decode storage w.1 addr with
  | none => rfl
  | some pair =>
    obtain ⟨member,inside,addrEq⟩ := decode_some h kind storage w.1 addr pair he
    have localOutside : ¬(pair.1∈Roles.outputs kind ∧ ∃ t : Fin w.1.span, ∃ j : Fin h.BK,
        pair.2=relative h w.1 w.2 t.val j.val) := by
      rintro ⟨_,t,j,eq⟩
      apply outside
      refine ⟨pair.1,member,t,j,?_⟩
      rw [addrEq,eq,relocated]
    change after pair.1 pair.2=before addr
    rw [frame pair.1 pair.2 localOutside]
    exact congrArg before addrEq.symm

theorem disjoint (h : Call) (kind : Kind) (storage : Placement h kind) (a b : Work h)
    (ha : a∈schedule h) (hb : b∈schedule h) (different : a≠b) (addr : Nat)
    (wa : footprint h kind storage a addr) : ¬footprint h kind storage b addr := by
  rintro ⟨s,hs,t',j',eb⟩
  obtain ⟨r,hr,t,j,ea⟩ := wa
  have offsets := (address_unique h kind storage r s hr hs _ _
    (offset_inside h a.1 (schedule_valid h a ha) a.2 t j)
    (offset_inside h b.1 (schedule_valid h b hb) b.2 t' j') (ea.symm.trans eb)).2
  exact different (Grid.work_unique h a b ha hb t t' j j' offsets)

#axiomsClean address_unique
#axiomsClean decode_address
#axiomsClean disjoint
end
end VeriTile.Bench.Repairs.DecayCumsum.PackedStorage
