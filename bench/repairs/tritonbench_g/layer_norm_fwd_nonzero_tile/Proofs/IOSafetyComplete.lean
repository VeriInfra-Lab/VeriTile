import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.IOSafetyStreams
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.IOSafetyTail

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.IOSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

theorem append_after {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionBounds)
    (first rest : List ComputeStmt) (s t : State α)
    (before : Safety.traceSafeList A bounds first s) (executed : run A first s=some t)
    (after : Safety.traceSafeList A bounds rest t) : Safety.traceSafeList A bounds (first++rest) s := by
  rw [Safety.traceSafeList_append]
  refine ⟨before,?_⟩
  intro u hu
  have same:u=t:=Option.some.inj (hu.symm.trans executed)
  subst u
  exact after

theorem complete {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α)
    (bounds : RegionBounds) (access : FlatDomain.RowBounds f r stride M N (s.pids 0) bounds) :
    FP.Structural.TraceSafe (engine A) bounds (repaired f r stride M N eps p block hb) s := by
  obtain ⟨s₁,h₁,hr₁⟩:=prefix_success A f r stride M N eps p block hb s
  obtain ⟨s₂,h₂,hr₂⟩:=firstStream_success A cmp f r stride M N eps p block hb (s.pids 0) s₁ hr₁
  obtain ⟨s₃,h₃,hr₃⟩:=secondInput_success A cmp f r stride M N eps p block hb (s.pids 0) s₂ hr₂
  obtain ⟨s₄,h₄,hr₄⟩:=residual_success A f r stride M N eps p block hb (s.pids 0) s₃ hr₃
  obtain ⟨s₅,h₅,hr₅⟩:=normalization_success A f r stride M N eps p block hb (s.pids 0) s₄ hr₄
  let code:=(repaired f r stride M N eps p block hb).surfaceBody
  change Safety.traceSafeList _ _ code s
  rw [←List.take_append_drop 10 code]
  apply append_after _ _ _ _ s s₁ (prefix_safe A f r stride M N eps p block hb s bounds access) h₁
  rw [←List.take_append_drop 2 (code.drop 10)]
  apply append_after _ _ _ _ s₁ s₂ (firstStream_safe A cmp f r stride M N eps p block hb (s.pids 0) s₁ hr₁ bounds access) h₂
  simp only [List.drop_drop,Nat.reduceAdd]
  rw [←List.take_append_drop 1 (code.drop 12)]
  apply append_after _ _ _ _ s₂ s₃ (secondInput_safe A cmp f r stride M N eps p block hb (s.pids 0) s₂ hr₂ bounds access) h₃
  simp only [List.drop_drop,Nat.reduceAdd]
  rw [←List.take_append_drop 2 (code.drop 13)]
  apply append_after _ _ _ _ s₃ s₄ (residual_safe A f r stride M N eps p block hb (s.pids 0) s₃ hr₃ bounds access) h₄
  simp only [List.drop_drop,Nat.reduceAdd]
  rw [←List.take_append_drop 3 (code.drop 15)]
  apply append_after _ _ _ _ s₄ s₅ (normalization_safe A f r stride M N eps p block hb (s.pids 0) s₄ hr₄ bounds access) h₅
  simpa only [List.drop_drop,Nat.reduceAdd] using
    affine_safe A f r stride M N eps p block hb (s.pids 0) s₅ hr₅ bounds access

#axiomsClean complete
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.IOSafety
