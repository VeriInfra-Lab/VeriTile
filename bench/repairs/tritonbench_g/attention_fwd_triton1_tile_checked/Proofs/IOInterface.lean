import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.IOProgramSafety
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.IOResult

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem windows [Inhabited α] (ops : Ops α) (literal : ℝ→α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (values : Values α c)
    (bounds : RegionName→Nat) (bounded : (io ops literal c scale storeFlag ifcond).InBounds bounds values) :
    SourceSafety.Bounds c storeFlag bounds := by
  constructor
  · intro r member off active
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl
    · exact bounded.2.1 (0 : Fin 3) (⟨off,active⟩,PUnit.unit) trivial
    · exact bounded.2.1 (1 : Fin 3) (⟨off,active⟩,PUnit.unit) trivial
    · exact bounded.2.1 (2 : Fin 3) (⟨off,active⟩,PUnit.unit) trivial
    · exact bounded.2.2.1 (0 : Fin 2) (⟨off,active⟩,PUnit.unit) trivial
  · intro enabled off active
    exact bounded.2.2.1 (1 : Fin 2) (⟨off,active⟩,PUnit.unit) enabled

theorem mapped_safe [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (bounds : RegionName→Nat) (bounded : SourceSafety.Bounds c storeFlag bounds)
    (heads : List Nat) (valid : ∀ h∈heads,h<c.heads) (s : State (Term α)) :
    ScheduleSafe bounds (heads.map (invocation α c scale storeFlag ifcond)) s := by
  induction heads generalizing s with
  | nil => trivial
  | cons h hs ih =>
    refine ⟨?_,fun t _=>ih (fun h member=>valid h (by simp [member])) t⟩
    exact SourceSafety.kernel_safe c scale h storeFlag ifcond _ (by simp [invocation,Invocation.initial,bindRegisters])
      bounds bounded (valid h (by simp))

theorem schedule_safe [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (bounds : RegionName→Nat) (bounded : SourceSafety.Bounds c storeFlag bounds)
    (s : State (Term α)) : ScheduleSafe bounds (schedule α c scale storeFlag ifcond) s := by
  unfold schedule
  split
  · trivial
  · exact mapped_safe c scale storeFlag ifcond bounds bounded (List.range c.heads)
      (by intro h member; exact List.mem_range.mp member) s

theorem implements [Inhabited α] (ops : Ops α) (literal : ℝ→α) (zero : literal 0=ops.zero)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    (io ops literal c scale storeFlag ifcond).Implements (reference ops literal c scale ifcond) := by
  apply ProgramIO.Implements.intro
  · intro values bounds bounded initial _
    exact schedule_safe c scale storeFlag ifcond bounds
      (windows ops literal c scale storeFlag ifcond values bounds bounded) initial
  · intro values initial pinned
    refine ⟨SharedGrid.result c scale storeFlag ifcond initial,?_,?_,?_⟩
    · change runSchedule (schedule α c scale storeFlag ifcond) initial=_
      rw [full_run]
      exact SharedGrid.run_success c scale storeFlag ifcond initial
    · exact output_binding ops literal zero c scale storeFlag ifcond values initial pinned
    · exact frame ops literal c scale storeFlag ifcond values initial

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO
