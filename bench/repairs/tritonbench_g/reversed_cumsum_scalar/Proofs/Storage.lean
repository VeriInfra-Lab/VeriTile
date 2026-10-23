import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Storage
open VeriTile.Triton
noncomputable section

/-- Ordinary successful FP32 allocations, including disjoint live storage. -/
structure Placement (h : Call) where
  backing : NativeIndex.Backing h
  inputBytes : backing.input.bytes=4
  outputBytes : backing.output.bytes=4
  disjoint : backing.input.base+backing.input.extent*4≤backing.output.base ∨
    backing.output.base+backing.output.extent*4≤backing.input.base

def address (buffer : NativeIndex.Buffer) (offset : Nat) := buffer.base+offset*buffer.bytes

def decode (buffer : NativeIndex.Buffer) (addr : Nat) : Option Nat :=
  if h : ∃ off, off<buffer.extent ∧ addr=address buffer off then some h.choose else none

theorem address_injective (buffer : NativeIndex.Buffer) : Function.Injective (address buffer) := by
  intro a b eq
  have hp := buffer.bytesPositive
  unfold address at eq
  nlinarith

theorem decode_address (buffer : NativeIndex.Buffer) (off : Nat) (inside : off<buffer.extent) :
    decode buffer (address buffer off)=some off := by
  unfold decode
  have hex : ∃ n, n<buffer.extent ∧ address buffer off=address buffer n := ⟨off,inside,rfl⟩
  rw [dif_pos hex]
  have eq := hex.choose_spec.2
  exact congrArg some ((address_injective buffer eq).symm)

theorem decode_some (buffer : NativeIndex.Buffer) (addr off : Nat) (he : decode buffer addr=some off) :
    off<buffer.extent ∧ addr=address buffer off := by
  unfold decode at he
  split at he
  · rename_i hex
    cases he
    exact hex.choose_spec
  · cases he

theorem input_not_output (h : Call) (p : Placement h) (off : Nat) (inside : off<p.backing.input.extent) :
    decode p.backing.output (address p.backing.input off)=none := by
  unfold decode
  rw [dif_neg]
  rintro ⟨other,ho,eq⟩
  have hd := p.disjoint
  dsimp only [address] at eq
  rw [p.inputBytes,p.outputBytes] at eq
  rcases hd with hd | hd <;> omega

def lift {β : Type} (h : Call) (p : Placement h) (other : β) (heap : Nat → β) : RegionName → Nat → β :=
  fun r off => if r=h.input then heap (address p.backing.input off)
    else if r=h.output then heap (address p.backing.output off) else other

def commit {β : Type} (h : Call) (p : Placement h) (before : Nat → β)
    (after : RegionName → Nat → β) : Nat → β :=
  fun addr => match decode p.backing.output addr with
    | none => before addr
    | some off => after h.output off

theorem lift_input {β : Type} (h : Call) (p : Placement h) (other : β) (heap : Nat → β) (off : Nat) :
    lift h p other heap h.input off=heap (address p.backing.input off) := by simp [lift]

theorem lift_output {β : Type} (h : Call) (layout : Geometry.OutputLayout h) (p : Placement h)
    (other : β) (heap : Nat → β) (off : Nat) :
    lift h p other heap h.output off=heap (address p.backing.output off) := by
  simp [lift,Ne.symm layout.distinct]

theorem commit_output {β : Type} (h : Call) (p : Placement h) (before : Nat → β)
    (after : RegionName → Nat → β) (off : Nat) (inside : off<p.backing.output.extent) :
    commit h p before after (address p.backing.output off)=after h.output off := by
  simp [commit,decode_address _ off inside]

theorem commit_input {β : Type} (h : Call) (p : Placement h) (before : Nat → β)
    (after : RegionName → Nat → β) (off : Nat) (inside : off<p.backing.input.extent) :
    commit h p before after (address p.backing.input off)=before (address p.backing.input off) := by
  simp [commit,input_not_output h p off inside]

theorem commit_initial {β : Type} (h : Call) (layout : Geometry.OutputLayout h) (p : Placement h)
    (other : β) (heap : Nat → β) : commit h p heap (lift h p other heap)=heap := by
  funext addr
  unfold commit
  cases eq : decode p.backing.output addr with
  | none => rfl
  | some off =>
    have spec := decode_some _ _ _ eq
    change lift h p other heap h.output off=heap addr
    rw [lift_output h layout p other heap off,←spec.2]

/-- A single source store is exactly one physical store through the ordinary
pointer adapter; cells not written by the source retain their prior contents. -/
theorem commit_write {β : Type} (h : Call) (p : Placement h) (before : Nat → β)
    (after : RegionName → Nat → β) (off : Nat) (inside : off<p.backing.output.extent) (value : β) :
    commit h p before (fun r k => if r=h.output ∧ k=off then value else after r k)=
      (fun addr => if addr=address p.backing.output off then value else commit h p before after addr) := by
  funext addr
  by_cases same : addr=address p.backing.output off
  · subst addr
    simp [commit,decode_address _ off inside]
  · unfold commit
    cases eq : decode p.backing.output addr with
    | none => simp [same]
    | some k =>
      have spec := decode_some _ _ _ eq
      have ne : k≠off := by intro hk; subst k; exact same spec.2
      simp [same,ne]

def writes (h : Call) (p : Placement h) (rows : List (Fin (h.B*h.H))) (addr : Nat) : Prop :=
  ∃ row∈rows, ∃ time, time<h.T ∧ addr=address p.backing.output (h.outputStrides.address h.H row.val time)

#print axioms commit_write
#print axioms input_not_output
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Storage
