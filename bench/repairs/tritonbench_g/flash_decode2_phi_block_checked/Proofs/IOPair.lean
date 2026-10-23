import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.PairIO
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.SourcePair
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.IOPair
open VeriTile Triton FP.Structural FlashDecode2
noncomputable section





private theorem map_pair {ι : Type} (left right : ι → Invocation Native.Term)
    (same : ∀ i s, (left i).run s = (right i).run s)
    (safe : ∀ i s bounds, (left i).Safe bounds s ↔ (right i).Safe bounds s)
    (ids : List ι) (s : State Native.Term) :
    runSchedule (ids.map left) s = runSchedule (ids.map right) s ∧
    ∀ bounds, ScheduleSafe bounds (ids.map left) s ↔ ScheduleSafe bounds (ids.map right) s := by
  induction ids generalizing s with
  | nil => exact ⟨rfl,fun _ => Iff.rfl⟩
  | cons i ids ih =>
    constructor
    · simp only [List.map_cons,runSchedule,same]
      congr 1
      funext t
      exact (ih t).1
    · intro bounds
      simp only [List.map_cons,ScheduleSafe,safe,same]
      exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => (ih t).2 bounds)

theorem schedules (g : PhiIO.Geometry) (order : List Nat) (s : State Native.Term) :
    runSchedule (PhiIO.schedule Native.algebra g order) s = runSchedule (order.map (job g)) s ∧
    ∀ bounds, ScheduleSafe bounds (PhiIO.schedule Native.algebra g order) s ↔
      ScheduleSafe bounds (order.map (job g)) s := by
  exact map_pair (PhiIO.launch Native.algebra g) (job g)
    (fun row s => SourcePair.source_exec Native.algebra g.H g.S g.D g.Q g.BD _)
    (fun row s bounds => SourcePair.source_safe Native.algebra g.H g.S g.D g.Q g.BD _ bounds) order s

/-- Existing complete native IO provides actual success, outputs and frame;
the exact integer-address equality transports the complete grid to the candidate. -/
theorem equivalent (g : PhiIO.Geometry) (width : g.D ≤ g.BD) (order : List Nat)
    (coverage : order.Perm (List.range (g.B * g.H))) :
    (io g order).EquivalentOn (variant g order) (PhiIO.Fits g) := by
  intro values domain bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := NativeIO.implements g width order coverage values domain bounds hb s hin
  have same := schedules g order s
  refine ⟨safe,(same.2 bounds).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · exact same.1.symm.trans ran
  · intro o j active
    rfl
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.IOPair
