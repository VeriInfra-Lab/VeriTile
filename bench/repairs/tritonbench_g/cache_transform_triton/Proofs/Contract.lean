import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.CacheTransform
open Lean VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 5000

def pairGather {α : Type} [Inhabited α] (M : Algebra α) (cos sin : Parameters)
    (s : State α) : State α := gather M sin (gather M cos s)

/-- Both fresh outputs are read back from independent direct input views. The
read-only inputs may overlap. Allocator freshness is
separate from numerical assumptions and entails these region inequalities. -/
theorem pair_correct {α : Type} [Inhabited α] (M : Algebra α)
    (cos sin : Parameters) (s : State α) (n cosCapacity sinCapacity : Nat)
    (rows cosBytes sinBytes : Nat → Nat)
    (hc : 0 < cos.count → GatherInput cos s n cosCapacity rows cosBytes)
    (hs : 0 < sin.count → GatherInput sin s n sinCapacity rows sinBytes)
    (freshOutput : cos.output ≠ sin.output)
    (freshCos : sin.cache ≠ cos.output) (freshRows : sin.rows ≠ cos.output) :
    (∀ o, o < cos.count → (pairGather M cos sin s).mem cos.output o =
      .mk .nat (byteTarget cos rows cosBytes o)) ∧
    (∀ o, o < sin.count → (pairGather M cos sin s).mem sin.output o =
      .mk .nat (byteTarget sin rows sinBytes o)) ∧
    (∀ r o, (r ≠ cos.output ∨ o ≥ cos.count) → (r ≠ sin.output ∨ o ≥ sin.count) →
      (pairGather M cos sin s).mem r o = s.mem r o) := by
  have hc' := gather_correct_total M cos s n cosCapacity rows cosBytes hc
  have hs' : 0 < sin.count → GatherInput sin (gather M cos s) n sinCapacity rows sinBytes := by
    intro hpos
    have hi := hs hpos
    refine { hi with table := ?_, payload := ?_ }
    · intro q hq
      rw [hc'.2 sin.rows q (Or.inl freshRows)]
      exact hi.table q hq
    · intro q hq
      rw [hc'.2 sin.cache q (Or.inl freshCos)]
      exact hi.payload q hq
  have hout := gather_correct_total M sin (gather M cos s) n sinCapacity rows sinBytes hs'
  refine ⟨?_, hout.1, ?_⟩
  · intro o ho
    rw [pairGather, hout.2 cos.output o (Or.inl freshOutput), hc'.1 o ho]
  · intro r o h₁ h₂
    rw [pairGather, hout.2 r o h₂, hc'.2 r o h₁]

/-- Row/byte coordinates decode the flat output, with no byte arithmetic in
the independent sequence-level target. -/
theorem byteTarget_coordinate (p : Parameters) (rows cache : Nat → Nat)
    (r c : Nat) (hw : 0 < p.rowBytes) (hc : c < p.rowBytes) :
    byteTarget p rows cache (r * p.rowBytes + c) = cache (rows r * p.rowBytes + c) := by
  have hd : (r * p.rowBytes + c) / p.rowBytes = r := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hw, Nat.div_eq_of_lt hc, Nat.zero_add]
  have hm : (r * p.rowBytes + c) % p.rowBytes = c := by
    rw [Nat.mul_add_mod', Nat.mod_eq_of_lt hc]
  simp only [byteTarget, hd, hm]

/-- Every stored element is reconstructed from exactly its own bytes. Any
decoder/observer is allowed, including native bit words and a Real observer. -/
theorem word_observation {α β : Type} [Inhabited α] (p : Parameters)
    (t : State α) (rows cache : Nat → Nat) (n width itemBytes : Nat)
    (hshape : p.count = n * (width * itemBytes))
    (hwidth : p.rowBytes = width * itemBytes)
    (hb : 0 < itemBytes)
    (hout : ∀ o, o < p.count → t.mem p.output o = .mk .nat (byteTarget p rows cache o))
    (observe : (Fin itemBytes → Nat) → β) (r col : Nat) (hr : r < n) (hc : col < width) :
    observe (fun b => (t.mem p.output ((r * width + col) * itemBytes + b.val)).read .nat) =
      observe (fun b => cache ((rows r * width + col) * itemBytes + b.val)) := by
  congr 1
  funext b
  obtain ⟨hcoord, hcol⟩ := stored_word_coordinate r col b.val width itemBytes hc b.isLt hb
  have hrow : 0 < p.rowBytes := by rw [hwidth]; exact Nat.mul_pos (by omega) hb
  have ho : (r * width + col) * itemBytes + b.val < p.count := by
    rw [hshape, hcoord]
    calc
      r * (width * itemBytes) + (col * itemBytes + b.val) <
          (r + 1) * (width * itemBytes) := by simpa [Nat.add_mul] using Nat.add_lt_add_left hcol (r * (width * itemBytes))
      _ ≤ n * (width * itemBytes) := Nat.mul_le_mul_right _ hr
  rw [hout _ ho, Cell.read_same, hcoord, ← hwidth,
    byteTarget_coordinate p rows cache r (col * itemBytes + b.val) hrow (hwidth ▸ hcol)]
  congr 1
  rw [hwidth]
  ring

/-- Logical Tensor coordinates and backing storage metadata. Admission selects
contiguous FP32 views without unresolved negative or conjugate flags. Read-only
inputs may share storage. -/
structure LogicalView (α : Type) where
  (rows width : Nat)
  (logical : Nat → Nat → α)
  (storageId offset rowStride colStride device : Nat)
  (dtype : Storage)

/-- A device context restores the caller's active device on every exit,
including validation/setup/launch failures. This is the ordinary context
manager contract; kernel and host code have no other device-state writes. -/
inductive Outcome (α : Type) where
  | returned (value : α)
  | failed (message : String)

def withDevice {α : Type} (caller selected : Nat) (body : Nat → Outcome α) : Nat × Outcome α :=
  (caller, body selected)

theorem device_restored {α : Type} (caller selected : Nat) (body : Nat → Outcome α) :
    (withDevice caller selected body).1 = caller := rfl

#axiomsClean pair_correct
#axiomsClean word_observation
end
end VeriTile.Bench.Repairs.CacheTransform
