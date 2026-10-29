import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Source

namespace VeriTile.Bench.Repairs.FastCeLoss.Grid
open Source CrossEntropy1
noncomputable section
set_option maxHeartbeats 3000000

structure ForwardSeparated (b : Buffers) : Prop where
  chunksLosses : b.chunks.storage≠b.losses.storage
  chunksLogits : b.chunks.storage≠b.logits.storage
  lossesLogits : b.losses.storage≠b.logits.storage

theorem row_major_injective {n row col otherRow otherCol : Nat}
    (colBound : col<n) (otherBound : otherCol<n)
    (equal : row*n+col=otherRow*n+otherCol) : row=otherRow ∧ col=otherCol := by
  have rows : row=otherRow := by
    by_contra h
    rcases lt_or_gt_of_ne h with h|h
    · have bound := Nat.mul_le_mul_right n (Nat.succ_le_of_lt h)
      nlinarith
    · have bound := Nat.mul_le_mul_right n (Nat.succ_le_of_lt h)
      nlinarith
  exact ⟨rows,by simpa only [rows,Nat.add_left_cancel_iff] using equal⟩

theorem forward_values_stable {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry current : Memory α)
    (same : ∀ p, p.1=b.logits.storage → current p=entry p) (q : Task) :
    forwardValues a o s b labels current q=forwardValues a o s b labels entry q := by
  have eq (i : Nat) : current (addr b.logits i)=entry (addr b.logits i) := same _ rfl
  have lanes : tile a o s b current q=tile a o s b entry q := by
    funext lane
    simp only [tile,eq]
  funext point
  simp only [forwardValues,forwardLoss,forwardLSE,lanes,target,eq]

theorem forward_frozen {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (sep : ForwardSeparated b) :
    forwardGrid a o s b labels entry=
      Schedule.run (Schedule.put (forwardWrites s b) (forwardValues a o s b labels entry))
        (tasks s.rows s.chunks) entry := by
  apply Schedule.freeze_reads (reads:=fun _ p => p.1=b.logits.storage)
    (writes:=forwardWrites s b) (entry:=entry)
  · intro q current same
    funext point
    change (if forwardWrites s b q point then forwardValues a o s b labels current q point else current point) =
      (if forwardWrites s b q point then forwardValues a o s b labels entry q point else current point)
    rw [forward_values_stable a o s b labels entry current same q]
  · intro q current point outside
    exact if_neg outside
  · exact CrossEntropy1.Grid.task_nodup _ _
  · intro q _ other _ _ point input writes
    rcases writes with h|⟨_,h⟩
    · have hreg := congrArg Prod.fst h
      exact sep.chunksLogits (hreg.symm.trans input)
    · have hreg := congrArg Prod.fst h
      exact sep.lossesLogits (hreg.symm.trans input)
  · intro _ _ _ _
    rfl

theorem chunks_address_distinct (b : Buffers) (sep : ForwardSeparated b) (i j : Nat) :
    addr b.chunks i ≠ addr b.losses j := by
  intro h
  exact sep.chunksLosses (congrArg Prod.fst h)

theorem forward_lse {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (sep : ForwardSeparated b)
    (row chunk : Nat) (hr : row<s.rows) (hc : chunk<s.chunks) :
    forwardGrid a o s b labels entry (addr b.chunks (row*s.chunks+chunk))=
      forwardLSE a o s b entry (row,chunk) := by
  rw [forward_frozen a o s b labels entry sep]
  have present : (row,chunk)∈tasks s.rows s.chunks := CrossEntropy1.Grid.task_mem _ _ _ |>.mpr ⟨hr,hc⟩
  have unique : ∀ q∈tasks s.rows s.chunks,
      forwardWrites s b q (addr b.chunks (row*s.chunks+chunk)) → q=(row,chunk) := by
    intro q hq hw
    have bounds := (CrossEntropy1.Grid.task_mem _ _ q).mp hq
    rcases hw with he|⟨_,he⟩
    · have offset := congrArg Prod.snd he
      have equal : row*s.chunks+chunk=q.1*s.chunks+q.2 := by simpa [addr] using offset
      rcases row_major_injective hc bounds.2 equal with ⟨r,c⟩
      exact Prod.ext r.symm c.symm
    · exact False.elim (chunks_address_distinct b sep _ _ he)
  rw [Schedule.fixed_output (forwardWrites s b) (forwardValues a o s b labels entry)
    (tasks s.rows s.chunks) entry (row,chunk) _ (Or.inl rfl) unique,if_pos present]
  simp only [forwardValues,if_neg (chunks_address_distinct b sep _ _)]

theorem forward_loss {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (sep : ForwardSeparated b)
    (row : Nat) (hr : row<s.rows) (positive : 0<s.chunks) :
    forwardGrid a o s b labels entry (addr b.losses row)=
      forwardLoss a o s b labels entry (row,0) := by
  rw [forward_frozen a o s b labels entry sep]
  have present : (row,0)∈tasks s.rows s.chunks := CrossEntropy1.Grid.task_mem _ _ _ |>.mpr ⟨hr,positive⟩
  have unique : ∀ q∈tasks s.rows s.chunks,
      forwardWrites s b q (addr b.losses row) → q=(row,0) := by
    intro q _ hw
    rcases hw with he|⟨hz,he⟩
    · exact False.elim (chunks_address_distinct b sep _ _ he.symm)
    · have offset := congrArg Prod.snd he
      have equal : row=q.1 := by simpa [addr] using offset
      exact Prod.ext equal.symm hz
  rw [Schedule.fixed_output (forwardWrites s b) (forwardValues a o s b labels entry)
    (tasks s.rows s.chunks) entry (row,0) _ (Or.inr ⟨rfl,rfl⟩) unique,if_pos present]
  simp [forwardValues]

theorem forward_frame {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (sep : ForwardSeparated b)
    (point : Address) (outside : ∀ q∈tasks s.rows s.chunks, ¬forwardWrites s b q point) :
    forwardGrid a o s b labels entry point=entry point := by
  rw [forward_frozen a o s b labels entry sep]
  exact Schedule.fixed_frame _ _ _ _ _ outside

#axiomsClean forward_frozen
#axiomsClean forward_lse
#axiomsClean forward_loss
#axiomsClean forward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Grid
