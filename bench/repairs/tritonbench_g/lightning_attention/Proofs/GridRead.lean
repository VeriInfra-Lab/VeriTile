import bench.repairs.tritonbench_g.lightning_attention.Proofs.Ownership
import bench.repairs.tritonbench_g.lightning_attention.Proofs.Readback
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem grid_nodup (stage : Stage) (d : Host.Dimensions) : (grid stage d).Nodup := by
  have he : grid stage d=
      ((List.range (Host.grid d stage).1).product (List.range (Host.grid d stage).2.1)).map
        (fun pair => (⟨pair.1,pair.2⟩ : PID)) := by
    simp [grid,List.product,List.map_flatMap,List.map_map,Function.comp_def]
  rw [he]
  apply List.Nodup.map
  · intro a b he
    cases a; cases b
    cases he
    rfl
  · exact List.nodup_range.product List.nodup_range

/-- A grid reads its owner's predecessor cell exactly once. This supports the
inter stage's additive current-memory update without replacing it by a supplied
gradient value or an immutable memory snapshot. -/
theorem grid_transform (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (hc : Host.Contract d) (schedule : List PID) (input current : Memory α)
    (hin : Inputs current input) (pid : PID) (r : RegionName) (o : Nat)
    (transform : Cell α → Cell α)
    (valid : ∀ x∈schedule, ValidPID stage d x) (unique : schedule.Nodup) (hm : pid∈schedule)
    (hw : Footprint stage d pid r o)
    (hit : ∀ mem, Inputs mem input → stageMemory M D stage d pid mem r o=transform (mem r o)) :
    gridMemory M D stage d schedule current r o=transform (current r o) := by
  classical
  induction schedule generalizing current with
  | nil => simp at hm
  | cons first rest ih =>
    have hp := valid pid hm
    have hfirst := valid first (by simp)
    have hrest : ∀ x∈rest, ValidPID stage d x := fun x hx => valid x (by simp [hx])
    have hn := List.nodup_cons.mp unique
    change gridMemory M D stage d rest (stageMemory M D stage d first current) r o=_
    by_cases he : first=pid
    · subst first
      rw [grid_frame]
      · exact hit current hin
      · intro other ho hw'
        have eq := footprint_unique stage d hc other pid (hrest other ho) hp r o hw' hw
        exact hn.1 (eq ▸ ho)
    · have hmrest : pid∈rest := by
        rcases List.mem_cons.mp hm with heq|hmrest
        · exact False.elim (he heq.symm)
        · exact hmrest
      have tail := ih (stageMemory M D stage d first current)
        ((stage_inputs M D stage d first current).trans hin) hrest hn.2 hmrest
      rw [tail]
      rw [stage_frame M D stage d first current r o]
      intro hw'
      exact he (footprint_unique stage d hc first pid hfirst hp r o hw' hw)

theorem full_grid_transform (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (pid : PID) (hp : ValidPID stage d pid) (r : RegionName) (o : Nat)
    (transform : Cell α → Cell α) (hw : Footprint stage d pid r o)
    (hit : ∀ mem, Inputs mem input → stageMemory M D stage d pid mem r o=transform (mem r o)) :
    gridMemory M D stage d (grid stage d) current r o=transform (current r o) :=
  grid_transform M D stage d hc _ input current hin pid r o transform
    (fun x hx => (mem_grid stage d x).mp hx) (grid_nodup stage d) ((mem_grid stage d pid).mpr hp) hw hit

theorem modelThree_exterior (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (m : Memory α) (r : RegionName) (o : Nat)
    (hout : regionCapacity d r≤o) :
    modelThree M D d (grid .forward d) (grid .intra d) (grid .inter d) m r o=m r o := by
  have safe : ∀ stage pid, pid∈grid stage d → ¬Footprint stage d pid r o := by
    intro stage pid hp hw
    have hb := footprint_bound stage d hc pid ((mem_grid stage d pid).mp hp) r o hw
    omega
  unfold modelThree
  rw [grid_frame _ _ _ _ _ _ _ _ (safe .inter),
    grid_frame _ _ _ _ _ _ _ _ (safe .intra),grid_frame _ _ _ _ _ _ _ _ (safe .forward)]
#axiomsClean grid_nodup
#axiomsClean grid_transform
#axiomsClean full_grid_transform
#axiomsClean modelThree_exterior
end
end VeriTile.Bench.Repairs.LightningAttention
