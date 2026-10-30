import bench.tritonbench_g.chunk_cumsum_vector.Proofs.NativeIndex
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.Grid

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Storage
open VeriTile.Triton
noncomputable section

/-- Ordinary actual allocation windows and disjoint live input/output storage.
Widths are explicit; FP32 public wrappers specialize both widths to four. -/
structure Placement (h : Call) where
  backing : NativeIndex.Backing h
  disjoint : backing.input.base+backing.input.extent*backing.input.bytes≤backing.output.base ∨
    backing.output.base+backing.output.extent*backing.output.bytes≤backing.input.base

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
  exact congrArg some ((address_injective buffer hex.choose_spec.2).symm)

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
  have hp := p.backing.input.bytesPositive
  have hz := p.backing.output.bytesPositive
  have hi := Nat.mul_lt_mul_of_pos_right inside hp
  have ho' := Nat.mul_lt_mul_of_pos_right ho hz
  dsimp only [address] at eq
  rcases p.disjoint with hd | hd <;> omega

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

def writes (h : Call) (p : Placement h) (programs : List (Geometry.Program h)) (addr : Nat) : Prop :=
  ∃ prog∈programs, ∃ (time : Nat) (j : Fin BS), Grid.active h prog time j ∧
    addr=address p.backing.output (Grid.offset h prog time j)

theorem commit_frame {β : Type} (h : Call) (layout : Geometry.OutputLayout h) (p : Placement h)
    (other : β) (before : Nat → β) (after : RegionName → Nat → β) (prog : Geometry.Program h)
    (frame : ∀ r off, ¬Grid.footprint h prog (r,off) → lift h p other before r off=after r off)
    (addr : Nat) (outside : ¬writes h p [prog] addr) : commit h p before after addr=before addr := by
  unfold commit
  cases eq : decode p.backing.output addr with
  | none => rfl
  | some off =>
    have spec := decode_some _ _ _ eq
    have out : ¬Grid.footprint h prog (h.output,off) := by
      rintro ⟨_,time,j,ha,he⟩
      dsimp only at he
      exact outside ⟨prog,by simp,time,j,ha,spec.2.trans (congrArg (address p.backing.output) he)⟩
    change after h.output off=before addr
    rw [←frame h.output off out,lift_output h layout p other before off,←spec.2]

#axiomsClean commit_write
#axiomsClean commit_frame
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.Storage
