import bench.tritonbench_g.f8_conversion_utils.Proofs.SourceReadback

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000

/-- Frame of the actual source state, including every inactive output cell
and every other region. The duplicate decode store uses the same frame. -/
theorem stores_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B pid : Nat) (snapshot s : State α) (r : RegionName) (i : Nat)
    (outside : r≠Y ∨ ∀ j : Fin B,pid*B+j.val<N → i≠pid*B+j.val) :
    (stores M p Y X N B pid snapshot s).mem r i=s.mem r i := by
  unfold stores
  apply State.masked_scatter_frame
  rcases outside with region|cell
  · exact Or.inl region
  · exact Or.inr (fun j _ active => cell j.1 active)

theorem memory_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B : Nat) (s : State α) (r : RegionName) (i : Nat)
    (outside : r≠Y ∨ ∀ j : Fin B,s.pids 0*B+j.val<N → i≠s.pids 0*B+j.val) :
    memory M p Y X N B s r i=s.mem r i := by
  unfold memory
  split
  · exact (stores_frame M p Y X N B (s.pids 0) s _ r i outside).trans
      (stores_frame M p Y X N B (s.pids 0) s s r i outside)
  · exact stores_frame M p Y X N B (s.pids 0) s s r i outside

#axiomsClean memory_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source
