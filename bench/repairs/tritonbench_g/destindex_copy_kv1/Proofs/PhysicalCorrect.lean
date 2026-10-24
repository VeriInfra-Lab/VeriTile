import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Physical

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Physical
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

theorem correct (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before : Memory) (binding : TensorBinding m w before) (other : State Word)
    (ps : List Nat) (complete : ∀ p, p∈ps ↔ p<(config m).count) :
    RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => launch m ps before (addr (base m .output) x)) ∧
    (∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
      launch m ps before cell=before cell) := by
  let s := initial (fun x => before (addr (base m .source) x))
    (fun x => before (addr (base m .output) x)) (destination w) other
  have reads : Reads m w before before := ⟨fun _ _ => rfl,binding.metadata⟩
  have bound : Grid.Bound s (fun x => before (addr (base m .source) x)) (destination w) :=
    initial_bound _ _ _ _
  have link : Linked m s before := by
    intro x
    simp [s,initial,inputRegion,metadataRegion,outputRegion,Region.cast]
  have final := grid_linked rest m w h before before binding reads s bound link ps
    (fun p hp => (complete p).mp hp)
  unfold Linked at final
  have core := Grid.correct rest (config m) ps s _ _ bound h.unique complete
  simp_rw [final,Cell.read_same] at core
  have copies : RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => launch m ps before (addr (base m .output) x)) := by
    simpa only [s,initial_output] using core
  exact ⟨copies,grid_frame m w h before before binding reads ps (fun p hp => (complete p).mp hp)⟩

#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Physical
