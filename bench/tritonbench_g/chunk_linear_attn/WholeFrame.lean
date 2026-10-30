import bench.tritonbench_g.chunk_linear_attn.HostSum
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def writeRegions : Stage → List RegionName
  | .forwardState => ["history","final"]
  | .forwardOutput => ["output"]
  | .backwardState => ["historyGradient","initialGradient"]
  | .backwardGradient => ["queryGradient","keyGradient","valuePartials"]

theorem footprint_region (stage : Stage) (p : Parameters) (final : Bool) (pid : PID) (r : RegionName) (o : Nat)
    (h : Footprint stage p final pid r o) : r∈writeRegions stage := by
  cases stage with
  | forwardState =>
    rcases h with ⟨rfl,_⟩|⟨rfl,_⟩ <;> simp [writeRegions,historyName,finalName]
  | backwardState =>
    rcases h with ⟨rfl,_⟩|⟨rfl,_⟩ <;> simp [writeRegions,historyName,finalName]
  | forwardOutput =>
    rcases h with ⟨rfl,_⟩
    simp [writeRegions]
  | backwardGradient =>
    rcases h with ⟨hr,_⟩|⟨rfl,_⟩
    · rcases hr with rfl|rfl <;> simp [writeRegions]
    · simp [writeRegions]

theorem grid_region_frame (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory α) (r : RegionName)
    (h : r∉writeRegions stage) :
    ∀ o, gridMemory M D stage p scale seed final schedule m r o=m r o := by
  intro o
  apply grid_frame
  intro pid _ hw
  exact h (footprint_region stage p final pid r o hw)

theorem data_not_written (stage : Stage) (r : RegionName)
    (h : r∈(["query","key","value","incoming","initial","terminalGradient"] : List RegionName)) :
    r∉writeRegions stage := by
  cases stage <;> simp only [writeRegions,List.mem_cons,List.not_mem_nil,or_false] at *
  all_goals rcases h with rfl|rfl|rfl|rfl|rfl|rfl <;> decide

theorem grid_data (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory α) :
    StateInputs (gridMemory M D stage p scale seed final schedule m) m := by
  intro r hr
  exact grid_region_frame M D stage p scale seed final schedule m r (data_not_written stage r hr)

theorem four_data (M : Algebra α) (D : DotPrimitive α) (forward backward : Parameters)
    (scale : ℝ) (seed final terminal : Bool) (sf so sb sg : List PID) (m : Memory α) :
    StateInputs (modelFour M D forward backward scale seed final terminal sf so sb sg m) m := by
  unfold modelFour
  exact (grid_data M D .backwardGradient backward scale Bool.false Bool.false sg _).trans
    ((grid_data M D .backwardState backward scale terminal seed sb _).trans
      ((grid_data M D .forwardOutput forward scale Bool.false Bool.false so _).trans
        (grid_data M D .forwardState forward scale seed final sf m)))

theorem sum_data (M : Algebra α) (p : Parameters) (selection : LibrarySelection p) (input current : Memory α) :
    StateInputs (hostSumMemory M p selection input current) current := by
  intro r hr o
  apply host_sum_frame
  left
  intro he
  subst r
  simp at hr

#axiomsClean footprint_region
#axiomsClean grid_region_frame
#axiomsClean grid_data
#axiomsClean four_data
#axiomsClean sum_data
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
