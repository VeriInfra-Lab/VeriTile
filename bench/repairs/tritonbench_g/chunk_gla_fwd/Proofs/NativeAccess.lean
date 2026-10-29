import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeBounds

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeAccess
open VeriTile Triton FP.Structural
open FPStages (Stage)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

theorem all_safe (bounds : RegionName → Nat) (stages : List (FillStage α))
    (h : ∀ stage∈stages,∀ s,stage.Safe bounds s) (s : State α) : FillScheduleSafe bounds stages s := by
  induction stages generalizing s with
  | nil => trivial
  | cons first rest ih =>
    refine ⟨h first (by simp) s,?_⟩
    intro next _
    exact ih (fun stage member=>h stage (List.mem_cons_of_mem _ member)) next

theorem grid_member (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (bounds : RegionName → Nat) (h : NativeLaunch.Bounds stage bounds)
    (job : FillStage α) (member : job∈NativeLaunch.jobs M D stage) (s : State α) : job.Safe bounds s := by
  obtain ⟨idx,_,same⟩ := List.mem_map.mp member
  subst job
  exact NativeSafety.source_safe M D stage idx bounds h s

theorem fill_safe (M : Algebra α) (region : RegionName) (count : Nat)
    (bounds : RegionName → Nat) (h : count≤bounds region) (s : State α) :
    (FillStage.fill (NativeLaunch.zero M region count)).Safe bounds s := by
  intro i inside
  simpa only [NativeLaunch.zero,Nat.zero_add] using lt_of_lt_of_le inside h

theorem producer (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions)
    (scale : ℝ) (bounds : RegionName → Nat) (h : NativeBounds.Producer c bounds) (s : State α) :
    FillScheduleSafe bounds (NativeLaunch.producer M D c scale) s := by
  apply all_safe
  intro job member state
  simp only [NativeLaunch.producer,List.mem_cons] at member
  rcases member with rfl|member
  · exact fill_safe M "attention" (Host.attentionSize c) bounds h.attention state
  · by_cases empty : c.heads*c.T=0
    · simp only [if_pos empty,List.not_mem_nil] at member
    · simp only [if_neg empty,List.mem_append] at member
      rcases member with inter|rest
      · exact grid_member M D (.inter c.producer scale) bounds (NativeBounds.inter c bounds h scale) job inter state
      · by_cases small : c.K≤256
        · simp only [if_pos small] at rest
          exact grid_member M D (.intra c.producer c.intraBK scale) bounds
            (NativeBounds.intra c bounds h scale) job rest state
        · simp only [if_neg small,List.mem_cons,List.mem_append] at rest
          rcases rest with rfl|split|merge
          · exact fill_safe M "scratch" (Host.scratchSize c) bounds
              (NativeBounds.scratch c bounds h empty small) state
          · exact grid_member M D (.split c.split scale) bounds
              (NativeBounds.split c bounds h scale empty small) job split state
          · exact grid_member M D (.merge c.split) bounds
              (NativeBounds.merge c bounds h empty small) job merge state

theorem consumer (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions)
    (scale : ℝ) (bounds : RegionName → Nat) (h : NativeBounds.Consumer c bounds) (s : State α) :
    FillScheduleSafe bounds (NativeLaunch.consumer M D c scale) s := by
  apply all_safe
  intro job member state
  by_cases empty : Host.outputSize c=0
  · simp only [NativeLaunch.consumer,if_pos empty,List.not_mem_nil] at member
  · simp only [NativeLaunch.consumer,if_neg empty] at member
    exact grid_member M D (.output c.consumer scale) bounds
      (NativeBounds.output c bounds h scale) job member state

#axiomsClean producer
#axiomsClean consumer
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeAccess
