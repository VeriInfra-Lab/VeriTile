import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.Storage
open VeriTile.Triton Layout
noncomputable section

/-- Valid remaining FP32 windows measured from each actual data pointer. -/
structure Placement (c : Shape) where
  backing : NativeIndex.Backing c
  input : RegionName
  output : RegionName
  distinct : output ≠ input
  disjoint : backing.input.base+4*backing.input.cells≤backing.output.base ∨
    backing.output.base+4*backing.output.cells≤backing.input.base

def address (buffer : NativeIndex.Buffer) (offset : Nat) := buffer.base+offset*4

def decode (buffer : NativeIndex.Buffer) (addr : Nat) : Option Nat :=
  if h : ∃ off, off<buffer.cells ∧ addr=address buffer off then some h.choose else none

theorem address_injective (buffer : NativeIndex.Buffer) : Function.Injective (address buffer) := by
  intro a b eq
  unfold address at eq
  omega

theorem decode_address (buffer : NativeIndex.Buffer) (off : Nat) (inside : off<buffer.cells) :
    decode buffer (address buffer off)=some off := by
  unfold decode
  have hex : ∃ n, n<buffer.cells ∧ address buffer off=address buffer n := ⟨off,inside,rfl⟩
  rw [dif_pos hex]
  exact congrArg some ((address_injective buffer hex.choose_spec.2).symm)

theorem decode_some (buffer : NativeIndex.Buffer) (addr off : Nat) (he : decode buffer addr=some off) :
    off<buffer.cells ∧ addr=address buffer off := by
  unfold decode at he
  split at he
  · rename_i hex
    cases he
    exact hex.choose_spec
  · cases he

def lift {β : Type} (c : Shape) (p : Placement c) (other : β) (heap : Nat → β) : RegionName → Nat → β :=
  fun r off => if r=p.input then heap (address p.backing.input off)
    else if r=p.output then heap (address p.backing.output off) else other

def commit {β : Type} (c : Shape) (p : Placement c) (before : Nat → β)
    (after : RegionName → Nat → β) : Nat → β :=
  fun addr => match decode p.backing.output addr with
    | none => before addr
    | some off => after p.output off

theorem lift_input {β : Type} (c : Shape) (p : Placement c) (other : β) (heap : Nat → β) (off : Nat) :
    lift c p other heap p.input off=heap (address p.backing.input off) := by simp [lift]

theorem lift_output {β : Type} (c : Shape) (p : Placement c)
    (other : β) (heap : Nat → β) (off : Nat) :
    lift c p other heap p.output off=heap (address p.backing.output off) := by
  simp [lift,p.distinct]

theorem commit_output {β : Type} (c : Shape) (p : Placement c) (before : Nat → β)
    (after : RegionName → Nat → β) (off : Nat) (inside : off<p.backing.output.cells) :
    commit c p before after (address p.backing.output off)=after p.output off := by
  simp [commit,decode_address _ off inside]

theorem commit_initial {β : Type} (c : Shape) (p : Placement c)
    (other : β) (heap : Nat → β) : commit c p heap (lift c p other heap)=heap := by
  funext addr
  unfold commit
  cases eq : decode p.backing.output addr with
  | none => rfl
  | some off =>
    have spec := decode_some _ _ _ eq
    change lift c p other heap p.output off=heap addr
    rw [lift_output,←spec.2]

/-- Representation of a single source store as one actual physical store. -/
theorem commit_write {β : Type} (c : Shape) (p : Placement c) (before : Nat → β)
    (after : RegionName → Nat → β) (off : Nat) (inside : off<p.backing.output.cells) (value : β) :
    commit c p before (fun r k => if r=p.output ∧ k=off then value else after r k)=
      (fun addr => if addr=address p.backing.output off then value else commit c p before after addr) := by
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

def writes (c : Shape) (p : Placement c) (addr : Nat) : Prop :=
  ∃ row, row<c.B*c.H ∧ ∃ time, time<c.T ∧ addr=address p.backing.output (row*c.T+time)

theorem offset_bound (c : Shape) (row time : Nat) (hr : row<c.B*c.H) (ht : time<c.T) :
    row*c.T+time<c.B*c.H*c.T := by
  have next := Nat.mul_le_mul_right c.T (Nat.succ_le_of_lt hr)
  rw [Nat.succ_mul] at next
  omega

theorem input_not_written (c : Shape) (p : Placement c) (off : Nat) (inside : off<p.backing.input.cells) :
    ¬writes c p (address p.backing.input off) := by
  rintro ⟨row,hr,time,ht,eq⟩
  have bound := (offset_bound c row time hr ht).trans_le p.backing.outputCells
  have hd := p.disjoint
  unfold address at eq
  rcases hd with hd | hd <;> omega

theorem commit_frame {β : Type} (c : Shape) (p : Placement c)
    (other : β) (before : Nat → β) (after : RegionName → Nat → β)
    (frame : ∀ off, (∀ row, row<c.B*c.H → ∀ time, time<c.T → off≠row*c.T+time) →
      after p.output off=lift c p other before p.output off) :
    ∀ addr, ¬writes c p addr → commit c p before after addr=before addr := by
  intro addr outside
  unfold commit
  cases eq : decode p.backing.output addr with
  | none => rfl
  | some off =>
    have spec := decode_some _ _ _ eq
    change after p.output off=before addr
    rw [frame off (by
      intro row hr time ht he
      exact outside ⟨row,hr,time,ht,he ▸ spec.2⟩),lift_output,←spec.2]

#axiomsClean commit_write
#axiomsClean commit_frame
#axiomsClean input_not_written
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.Storage
