import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.IO
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.DirectReal

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IORealGrid
open VeriTile Triton Host PublicIO
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1600000

@[simp] theorem initial (c : Shape) (q : GridIndex (grid c)) (s : BlockState) :
    (realLaunch c q).initial s = s.withGridIndex q := by
  rfl

def reference (c : Shape) (scale : ℝ) : List (GridIndex (grid c)) → BlockState → BlockState
  | [], s => s
  | q::rest, s => reference c scale rest (DirectReal.frames c scale s q).final

theorem run_memory (c : Shape) (scale : ℝ) (jobs : List (GridIndex (grid c))) (s : BlockState) :
    ∃ t, runAlgorithmSchedule (DirectReal.kernel c scale) (jobs.map (realLaunch c)) s = some t ∧
      t.mem = (reference c scale jobs s).mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s, rfl, rfl⟩
  | cons q rest ih =>
    obtain ⟨t, ht, hm⟩ := ih (DirectReal.frames c scale s q).final
    refine ⟨t, ?_, hm⟩
    simpa only [List.map_cons, runAlgorithmSchedule, AlgorithmLaunch.run, initial,
      (DirectReal.frames c scale s q).h_exec, Option.bind_some] using ht

theorem frame (c : Shape) (scale : ℝ) (jobs : List (GridIndex (grid c))) (s : BlockState)
    (r : RegionName) (off : Nat)
    (miss : ∀ q ∈ jobs, ¬ Real.footprint (layout c) (launchIndex c q) regions (r,off)) :
    (reference c scale jobs s).mem r off = s.mem r off := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    rw [reference, ih _ (fun k hk => miss k (List.mem_cons_of_mem q hk))]
    exact ((DirectReal.frames c scale s q).h_writeWithin r off (miss q (by simp))).symm

theorem inputs_preserved (c : Shape) (scale : ℝ) (s : BlockState) (q : GridIndex (grid c)) :
    inputs regions (DirectReal.frames c scale s q).final = inputs regions s := by
  have hf := (DirectReal.frames c scale s q).h_writeWithin
  have h (r : RegionName) (hr : r≠"o") (off : Nat) :
      (DirectReal.frames c scale s q).final.readMem r off = s.readMem r off := by
    unfold BlockState.readMem
    rw [← hf r off (by intro hh; exact hr hh.1)]
    rfl
  unfold inputs
  congr 1 <;> funext off
  all_goals exact h _ (by decide) off

theorem readback (c : Shape) (scale : ℝ) (jobs : List (GridIndex (grid c)))
    (s : BlockState) (hn : jobs.Nodup) (q : GridIndex (grid c)) (hq : q∈jobs)
    (i : TileIndex [c.BT,c.BV]) (hi : active (layout c) (launchIndex c q) i.1.val i.2.1.val) :
    (reference c scale jobs s).readMem "o"
      (outputAddress (layout c) (launchIndex c q) i.1.val i.2.1.val) =
      expected (layout c) (launchIndex c q) (inputs regions s) scale i.1.val i.2.1.val := by
  induction jobs generalizing s with
  | nil => simp at hq
  | cons u rest ih =>
    let mid := (DirectReal.frames c scale s u).final
    obtain ⟨hu, hn⟩ := List.nodup_cons.mp hn
    by_cases hqu : q=u
    · subst q
      have hf := frame c scale rest mid "o"
        (outputAddress (layout c) (launchIndex c u) i.1.val i.2.1.val) (by
          intro q hq hw
          obtain ⟨_,j,hj,he⟩ := hw
          have heq := launchIndex_injective c (address_injective c _ _ i j hi hj he).1
          exact hu (heq ▸ hq))
      change (reference c scale rest mid).readMem _ _ = _
      unfold BlockState.readMem
      rw [hf]
      exact (DirectReal.perProgram c scale s u).choose_spec.2.1 i hi
    · have hqr : q∈rest := by simpa [hqu] using hq
      change (reference c scale rest mid).readMem _ _ = _
      rw [ih mid hn hqr, inputs_preserved]

theorem complete (c : Shape) (bt : 0<c.BT) (scale : ℝ) (s : BlockState) :
    ∃ t, runAlgorithmSchedule (DirectReal.kernel c scale) (realSchedule c) s = some t ∧
      (∀ h r v, t.readMem "o" (h.val*(c.T*c.V)+r.val*c.V+v.val) =
        Host.formula c bt (inputs regions s) scale h r v) ∧
      BlockState.WriteWithin (DirectReal.footprint c) s t := by
  obtain ⟨t, ht, hm⟩ := run_memory c scale (FPGrid.programs c) s
  refine ⟨t, ht, ?_, ?_⟩
  · intro h r v
    have hv := readback c scale (FPGrid.programs c) s (FPGrid.programs_nodup c)
      (owner c bt h r v) (FPGrid.mem_programs c _) (lane c bt r v) (owner_active c bt h r v)
    simpa only [BlockState.readMem, hm, owner_address, Host.formula] using hv
  · intro r off miss
    rw [hm]
    symm
    apply frame
    intro q _ hit
    exact miss (DirectReal.frames_bound c scale s _ ⟨q,hit⟩)

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IORealGrid
