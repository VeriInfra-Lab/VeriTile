import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Preparation

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 5000000

structure Frame {M α β : Type} (ops : Ops M α β) (caller : List Allocation)
    (before after : World M) : Prop where
  reads : ∀ old∈caller,∀ key,old.contains key → ops.read after.memory key=ops.read before.memory key
  bytes : ∀ old∈caller,∀ owner address,old.covers owner address →
    ops.observe after.memory owner address=ops.observe before.memory owner address

theorem Frame.refl {M α β : Type} (ops : Ops M α β) (old : List Allocation) (s : World M) : Frame ops old s s :=
  ⟨fun _ _ _ _ => rfl,fun _ _ _ _ _ => rfl⟩

theorem Frame.trans {M α β : Type} (ops : Ops M α β) {old : List Allocation} {s t u : World M}
    (h : Frame ops old s t) (k : Frame ops old t u) : Frame ops old s u :=
  ⟨fun a ha key hk => (k.reads a ha key hk).trans (h.reads a ha key hk),
    fun a ha owner address hk => (k.bytes a ha owner address hk).trans (h.bytes a ha owner address hk)⟩

theorem Keep.frame {M α β : Type} (ops : Ops M α β) {s t : World M} (h : Keep ops s t)
    (old : List Allocation) (live : ∀ a∈old,a∈s.allocations) : Frame ops old s t :=
  ⟨fun a ha => h.reads a (live a ha),fun a ha => h.bytes a (live a ha)⟩

theorem separate_read (old fresh : Allocation) (separate : fresh.separate old)
    (writer observed : Key) (hw : fresh.contains writer) (ho : old.contains observed) :
    observed.disjoint writer :=
  Key.disjoint_symm (Allocation.spans_separate fresh old separate writer observed hw ho)

theorem separate_byte (old fresh : Allocation) (separate : fresh.separate old)
    (writer : Key) (hw : fresh.contains writer) (owner address : Nat)
    (ho : old.covers owner address) : ¬writer.covers owner address := by
  rcases hw with ⟨wo,wlow,whigh⟩
  rcases ho with ⟨oo,olow,ohigh⟩
  rintro ⟨keyOwner,low,high⟩
  rcases separate with owners | before | after
  · exact owners (wo.symm.trans (keyOwner.symm.trans oo))
  · omega
  · omega

theorem private_run_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (valid : Valid p)
    (old : List Allocation) (fresh : ∀ a∈old,p.output.separate a)
    (schedule : List Nat) (bounded : ∀ row∈schedule,row<p.rows) (s : World M) :
    Frame ops old s {s with memory := Physical.run ops compute p schedule s.memory} := by
  constructor
  · intro a ha key hk
    apply run_frame
    intro row hr col hc
    exact separate_read a p.output (fresh a ha) _ _ (ybacked p valid row col (bounded row hr) hc) hk
  · intro a ha owner address haddr
    apply run_bytes
    intro row hr col hc
    exact separate_byte a p.output (fresh a ha) _ (ybacked p valid row col (bounded row hr) hc) _ _ haddr

theorem private_interrupted_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (valid : Valid p)
    (old : List Allocation) (fresh : ∀ a∈old,p.output.separate a)
    (completed : List Nat) (bounded : ∀ row∈completed,row<p.rows)
    (storesDone : List (Nat × Nat)) (partialBound : ∀ ij∈storesDone,ij.1<p.rows ∧ ij.2<p.cols)
    (s : World M) : Frame ops old s
      {s with memory := Physical.interrupted ops compute p completed storesDone s.memory} := by
  have whole := private_run_frame ops compute p valid old fresh completed bounded s
  constructor
  · intro a ha key hk
    unfold Physical.interrupted
    rw [read_frame ops _ _ _ _ _]
    · exact whole.reads a ha key hk
    · intro ij hij
      obtain ⟨hr,hc⟩ := partialBound ij hij
      exact separate_read a p.output (fresh a ha) _ _ (ybacked p valid ij.1 ij.2 hr hc) hk
  · intro a ha owner address haddr
    apply interrupted_bytes
    · intro row hr col hc
      exact separate_byte a p.output (fresh a ha) _ (ybacked p valid row col (bounded row hr) hc) _ _ haddr
    · intro ij hij
      obtain ⟨hr,hc⟩ := partialBound ij hij
      exact separate_byte a p.output (fresh a ha) _ (ybacked p valid ij.1 ij.2 hr hc) _ _ haddr

theorem Keep.rowInputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (p : Layout) (valid : Valid p) {s t : World M} (h : Keep ops s t)
    (live : p.input∈s.allocations) (row : Nat) (hr : row<p.rows) :
    rowInputs ops p t.memory row=rowInputs ops p s.memory row := by
  funext col
  unfold Physical.rowInputs
  split
  · rename_i hc
    exact h.reads p.input live _ (xbacked p valid row col hr hc)
  · rfl

#axiomsClean private_run_frame
#axiomsClean private_interrupted_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host
