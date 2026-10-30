import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Numerical

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
open VeriTile.Triton FP.Structural
open BmmOptimized Source Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem lane_enumeration (c : Config) :
    (TileShape.allIndices [c.tm,c.tn]).map (laneOf (c:=c))=lanes c := by
  have hm : (List.finRange c.tm).map Fin.val=List.range c.tm := List.map_coe_finRange_eq_range
  have hn : (List.finRange c.tn).map Fin.val=List.range c.tn := List.map_coe_finRange_eq_range
  change _=(List.range c.tm).flatMap (fun i=>(List.range c.tn).map (Prod.mk i))
  rw [←hm,←hn]
  simp only [TileShape.allIndices,laneOf,List.map_flatMap,List.flatMap_map,
    List.map_map,List.map_cons,List.map_nil,Function.comp_def]
  simp only [List.map_eq_flatMap]

def sourceWrites (o : Scalar α) (p : Parameters) (initial : State α) : List (Nat×α) :=
  (TileShape.allIndices [p.TM,p.TN]).filterMap fun i=>
    if Execution.active p initial i then some (oOffset p initial i,accumulation o p initial (iterations p) i) else none

theorem writes_equal (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (hp : pid∈programs s c) (m : Memory α) (state : State α) (related : Related m state) :
    sourceWrites o (parameters s c) (start s c pid state)=programWrites s c o pid m := by
  unfold sourceWrites programWrites
  rw [←lane_enumeration]
  simp only [List.map_map,List.filterMap_map]
  apply List.filterMap_congr
  intro i hi
  simp only [Function.comp_apply,active_exact s c pid hp state i,
    source_value s c o pid hp m state related i,o_offset s c pid state i]
  rfl

def commitState (writes : List (Nat×α)) (state : State α) : State α :=
  writes.foldl (fun st w=>st.write "O" w.1 (.mk .real w.2)) state

theorem related_write (m : Memory α) (state : State α) (related : Related m state)
    (address : Nat) (value : α) :
    Related (BmmOptimized.store m address value) (state.write "O" address (.mk .real value)) := by
  constructor
  · intro off
    simpa [State.write,BmmOptimized.store] using related.a off
  · intro off
    simpa [State.write,BmmOptimized.store] using related.b off
  · intro off
    by_cases same : off=address <;>
      simp [State.write,BmmOptimized.store,related.output,same]

theorem related_commit (writes : List (Nat×α)) (m : Memory α) (state : State α)
    (related : Related m state) : Related (commit writes m) (commitState writes state) := by
  induction writes generalizing m state with
  | nil=>exact related
  | cons w ws ih=>
    exact ih _ _ (related_write m state related w.1 w.2)

theorem commit_frame (writes : List (Nat×α)) (state : State α) (r : RegionName) (off : Nat)
    (outside : ∀ w∈writes,r≠"O" ∨ off≠w.1) :
    (commitState writes state).mem r off=state.mem r off := by
  induction writes generalizing state with
  | nil=>rfl
  | cons w ws ih=>
    change (commitState ws (state.write "O" w.1 (.mk .real w.2))).mem r off=_
    rw [ih _ (fun v hv=>outside v (by simp [hv]))]
    exact State.write_other _ _ _ _ _ _ (outside w (by simp))

#axiomsClean lane_enumeration
#axiomsClean writes_equal
#axiomsClean related_commit
#axiomsClean commit_frame
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Bridge
