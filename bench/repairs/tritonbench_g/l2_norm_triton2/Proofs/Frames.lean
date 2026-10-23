import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Preparation

namespace VeriTile.Bench.Repairs.L2NormTriton2.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormTriton1.Host (World Result Keep Dense Frame separate_read separate_byte)
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 5000000

theorem private_run_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (valid : Valid p)
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
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (valid : Valid p)
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

theorem keep_inputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (p : Layout) (valid : Valid p) {s t : World M} (h : Keep ops s t)
    (gradient : Bool) (live : p.inputFor gradient∈s.allocations) (row : Nat) (hr : row<p.rows) :
    rowInputs ops p t.memory row gradient=rowInputs ops p s.memory row gradient := by
  funext col
  unfold Physical.rowInputs
  split
  · rename_i hc
    exact h.reads (p.inputFor gradient) live _ (xbacked p valid gradient row col hr hc)
  · rfl

#axiomsClean private_run_frame
#axiomsClean private_interrupted_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Host
