import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.ForwardGrid

namespace VeriTile.Bench.Repairs.FastCeLoss.Grid
open Source CrossEntropy1
noncomputable section
set_option maxHeartbeats 3000000

structure HostSeparated (b : Buffers) : Prop extends ForwardSeparated b where
  lseChunks : b.lse.storage≠b.chunks.storage
  lseLosses : b.lse.storage≠b.losses.storage
  lseLogits : b.lse.storage≠b.logits.storage

def finishReads (b : Buffers) (row : Nat) (point : Address) : Prop :=
  point.1=b.chunks.storage ∨ point=addr b.losses row

theorem finish_values_stable {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (entry current : Memory α) (row : Nat)
    (same : ∀ p, finishReads b row p → current p=entry p) :
    finishValues a s b labels current row=finishValues a s b labels entry row := by
  have hc (i : Nat) : current (addr b.chunks i)=entry (addr b.chunks i) := same _ (Or.inl rfl)
  have hl : current (addr b.losses row)=entry (addr b.losses row) := same _ (Or.inr rfl)
  funext point
  simp only [finishValues,globalLSE,hc,hl]

theorem finish_frozen {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (entry : Memory α) (sep : HostSeparated b) :
    Schedule.run (finishStep a s b labels) (List.range s.rows) entry=
      Schedule.run (Schedule.put (finishWrites b) (finishValues a s b labels entry))
        (List.range s.rows) entry := by
  apply Schedule.freeze_reads (reads:=finishReads b) (writes:=finishWrites b) (entry:=entry)
  · intro row current same
    funext point
    change (if finishWrites b row point then finishValues a s b labels current row point else current point) =
      (if finishWrites b row point then finishValues a s b labels entry row point else current point)
    rw [finish_values_stable a s b labels entry current row same]
  · intro row current point outside
    exact if_neg outside
  · exact List.nodup_range
  · intro row _ other _ different point read write
    rcases read with hc|hl <;> rcases write with wl|wy
    · exact sep.lseChunks ((congrArg Prod.fst wl).symm.trans hc)
    · exact sep.chunksLosses ((congrArg Prod.fst wy).symm.trans hc).symm
    · exact sep.lseLosses ((congrArg Prod.fst wl).symm.trans (congrArg Prod.fst hl))
    · have eq := congrArg Prod.snd (wy.symm.trans hl)
      apply different
      simpa only [addr,Nat.add_left_cancel_iff] using eq
  · intro _ _ _ _
    rfl

theorem finish_unique (b : Buffers) (sep : HostSeparated b) (row other : Nat)
    (point : Address) (here : finishWrites b row point) (there : finishWrites b other point) :
    other=row := by
  rcases here with hl|hy <;> rcases there with ol|oy
  · have eq := congrArg Prod.snd (ol.symm.trans hl)
    simpa only [addr,Nat.add_left_cancel_iff] using eq
  · exact False.elim (sep.lseLosses (congrArg Prod.fst (hl.symm.trans oy)))
  · exact False.elim (sep.lseLosses (congrArg Prod.fst (ol.symm.trans hy)))
  · have eq := congrArg Prod.snd (oy.symm.trans hy)
    simpa only [addr,Nat.add_left_cancel_iff] using eq

theorem finish_output {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (entry : Memory α) (sep : HostSeparated b)
    (chunked : s.chunks≠1) (row : Nat) (hr : row<s.rows) (point : Address)
    (active : finishWrites b row point) :
    finishGrid a s b labels entry point=finishValues a s b labels entry row point := by
  rw [finishGrid,if_neg chunked,finish_frozen a s b labels entry sep]
  rw [Schedule.fixed_output (finishWrites b) (finishValues a s b labels entry)
    (List.range s.rows) entry row point active]
  · simp [hr]
  · intro other _ writes
    exact finish_unique b sep row other point active writes

theorem finish_frame {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (entry : Memory α) (sep : HostSeparated b)
    (point : Address) (outside : ∀ row<s.rows, ¬finishWrites b row point) :
    finishGrid a s b labels entry point=entry point := by
  unfold finishGrid
  split_ifs with h
  · rfl
  · rw [finish_frozen a s b labels entry sep]
    exact Schedule.fixed_frame _ _ _ _ _ (by intro row hr; exact outside row (List.mem_range.mp hr))

#axiomsClean finish_frozen
#axiomsClean finish_output
#axiomsClean finish_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Grid
