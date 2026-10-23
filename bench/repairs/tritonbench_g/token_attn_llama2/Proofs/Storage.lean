import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.HostBounds

/-! Finite caller-owned storage and physical separation. This is a view
adapter for supplied buffers, not an allocator/runtime implementation proof. -/
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

inductive Buffer where
  | query | key | locations | starts | lengths | output
  deriving DecidableEq

def extent (c : Call) : Buffer → Nat
  | .query => c.qSize | .key => c.kSize | .locations => c.locSize
  | .starts | .lengths => c.batches | .output => c.outSize

structure Placement (c : Call) where
  arena : Nat
  base : Buffer → Nat
  bounded : ∀ r, base r + extent c r ≤ arena
  /-- Read buffers may overlap one another when their actual cells agree.
  Output must be physically separate from every read buffer. -/
  outputApart : ∀ r, r ≠ .output → ∀ i<extent c r, ∀ o<c.outSize,
    base r+i ≠ base .output+o

def Placement.index {c : Call} (p : Placement c) (r : Buffer) (i : Fin (extent c r)) :
    Fin p.arena := ⟨p.base r+i.val,by have h := p.bounded r; omega⟩

structure FlatBinding {α : Type} (c : Call) (p : Placement c)
    (m : Metadata c) (x : Buffers α c) (flat : Fin p.arena → Cell α) : Prop where
  query : ∀ i : Fin c.qSize, flat (p.index .query i) = .mk .fp32 (x.q i)
  key : ∀ i : Fin c.kSize, flat (p.index .key i) = .mk .fp32 (x.k i)
  output : ∀ i : Fin c.outSize, flat (p.index .output i) = .mk .fp32 (x.out i)
  starts : ∀ i : Fin c.batches, flat (p.index .starts i) = .mk .int (m.starts i)
  lengths : ∀ i : Fin c.batches, flat (p.index .lengths i) = .mk .int (m.lengths i)
  locations : ∀ i : Fin c.locSize, flat (p.index .locations i) = .mk .int (m.locations i)

/-- Materialize only the caller's output view. All cells outside it retain
exactly their original typed contents. Unwritten output cells are framed below. -/
def flatten {α : Type} {c : Call} (p : Placement c) (flat : Fin p.arena → Cell α)
    (final : State α) (i : Fin p.arena) : Cell α :=
  if p.base .output ≤ i.val ∧ i.val < p.base .output+c.outSize then
    final.mem "Out" (i.val-p.base .output)
  else flat i

theorem flatten_output {α : Type} {c : Call} (p : Placement c)
    (flat : Fin p.arena → Cell α) (final : State α) (i : Fin c.outSize) :
    flatten p flat final (p.index .output i) = final.mem "Out" i.val := by
  simp [flatten,Placement.index,i.isLt]

theorem flatten_input {α : Type} {c : Call} (p : Placement c)
    (flat : Fin p.arena → Cell α) (final : State α) (r : Buffer) (hr : r ≠ .output)
    (i : Fin (extent c r)) : flatten p flat final (p.index r i) = flat (p.index r i) := by
  unfold flatten
  apply if_neg
  intro h
  have he := p.outputApart r hr i.val i.isLt
    (p.base r+i.val-p.base .output) (by dsimp [Placement.index] at h; omega)
  apply he
  dsimp [Placement.index] at h
  omega

variable {α : Type} [Inhabited α]

def FlatResult (c : Call) (p : Placement c) (m : Metadata c) (value : Nat → Nat → Nat → α)
    (before after : Fin p.arena → Cell α) : Prop :=
  (∀ b<c.batches, ∀ h<c.heads, ∀ t<m.len b,
    ∀ j : Fin p.arena, j.val=p.base .output+(h*c.outHead+(m.begin b+t)*c.outSeq) →
      after j = .mk .fp32 (value b h t)) ∧
  (∀ j : Fin p.arena,
    (∀ b<c.batches, ∀ h<c.heads, ∀ t<m.len b,
      j.val ≠ p.base .output+(h*c.outHead+(m.begin b+t)*c.outSeq)) →
    after j = before j)

theorem finite_result (c : Call) (p : Placement c) (m : Metadata c) (x : Buffers α c)
    (s final : State α) (flat : Fin p.arena → Cell α) (bound : FlatBinding c p m x flat)
    (valid : Valid c m) (value : Nat → Nat → Nat → α)
    (result : Result c m value (initial c m x s) final) :
    FlatResult c p m value flat (flatten p flat final) := by
  refine ⟨?_,?_⟩
  · intro b hb h hh t ht j hj
    have hs := (valid.starts ⟨b,hb⟩).2
    dsimp at hs
    have ho := valid.outputBounds h hh (m.begin b+t) (by omega)
    have he : j = p.index .output ⟨h*c.outHead+(m.begin b+t)*c.outSeq,ho⟩ := by
      apply Fin.ext; exact hj
    rw [he,flatten_output]
    exact result.1 b hb h hh t ht
  · intro j hj
    unfold flatten
    split_ifs with inside
    · have ho : j.val-p.base .output<c.outSize := by omega
      have hu : Unwritten c m "Out" (j.val-p.base .output) := by
        right
        intro b hb h hh t ht he
        apply hj b hb h hh t ht
        omega
      rw [result.2 "Out" _ hu]
      have indexEq : p.index .output ⟨j.val-p.base .output,ho⟩ = j := by
        apply Fin.ext
        dsimp [Placement.index]
        omega
      have hbind := bound.output ⟨j.val-p.base .output,ho⟩
      have hbind2 := (congrArg flat indexEq).symm.trans hbind
      simpa [initial,finiteRead,ho] using hbind2.symm
    · rfl

#axiomsClean flatten_input
#axiomsClean finite_result
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
