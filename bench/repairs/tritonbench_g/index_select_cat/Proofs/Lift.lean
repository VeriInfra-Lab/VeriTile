import bench.repairs.tritonbench_g.index_select_cat.Proofs.Scatter
namespace VeriTile.Bench.Repairs.IndexSelectCat.ProgramGrid
open VeriTile.Triton FP.Structural
noncomputable section
def sourceRegion : RegionName := ⟨"index_select_source"⟩
def outputRegion : RegionName := ⟨"index_select_output"⟩
def liftOutput {α : Type} [Inhabited α] (base : Nat) (before : Nat → α)
    (s : State α) : Nat → α :=
  fun q => if base≤q then (s.mem outputRegion (q-base)).read .real else before q

theorem lift_write {α : Type} [Inhabited α] (base : Nat) (before : Nat → α)
    (s : State α) (offset : Nat) (v : α) :
    liftOutput base before (s.write outputRegion offset (.mk .real v)) =
      Scatter.write (liftOutput base before s) (base+offset) v := by
  funext q
  by_cases h : base≤q
  · have he : q-base=offset ↔ q=base+offset := by omega
    by_cases hit : q=base+offset <;> simp [liftOutput,Scatter.write,State.write,h,he,hit]
  · have hn : q≠base+offset := by omega
    simp [liftOutput,Scatter.write,h,hn]

theorem lift_scatter {α ι : Type} [Inhabited α] (base : Nat) (before : Nat → α)
    (s : State α) (events : List ι) (offset : ι → Nat) (val : ι → α) :
    liftOutput base before
      (events.foldl (fun state i => state.write outputRegion (offset i) (.mk .real (val i))) s) =
    Scatter.run events (fun i => base+offset i) val (liftOutput base before s) := by
  induction events generalizing s with
  | nil => rfl
  | cons i tail ih =>
    simp only [List.foldl_cons,ih,lift_write,Scatter.run,List.foldl_cons]


end
end VeriTile.Bench.Repairs.IndexSelectCat.ProgramGrid
