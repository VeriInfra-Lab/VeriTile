import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableSafety
import bench.tritonbench_g.layer_norm_welfold.IO
import bench.tritonbench_g.layer_norm_welfold.Proofs.KernelBoundary
import bench.tritonbench_g.layer_norm_welfold.Proofs.IOImplements
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem schedule_eq (M : Algebra α) (reuse : Bool) (Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) (rows : List Nat) (s : State α) :
    runSchedule (rows.map (PairIO.invocation M reuse Y Mean Rstd S D B tree)) s =
      SourceCode.runRows (withEquality M (fun _ _ => Bool.false))
        (SourceCode.selected reuse "X" "W" "Bias" Y Mean Rstd S D B (Readable.tree tree)) rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    simp only [List.map_cons,runSchedule,SourceCode.runRows]
    change (do
      let t ← exec (withEquality M (fun _ _ => Bool.false))
        (SourceCode.selected reuse "X" "W" "Bias" Y Mean Rstd S D B (Readable.tree tree)) (atRow s row)
      runSchedule (rows.map (PairIO.invocation M reuse Y Mean Rstd S D B tree)) t) = _
    congr 1
    funext t
    exact ih t

theorem schedule_safe (M : Algebra α) (observe : α → α) (reuse : Bool)
    (Y Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B)
    (separated : ("X" : RegionName) ≠ Mean)
    (vals : ChannelValues α (IO.inputs S D)) (bounds : RegionName → Nat)
    (bounded : (PairIO.layerNormIO M observe reuse Y Mean Rstd S D B tree).InBounds bounds vals)
    (rows : List Nat) (legal : ∀ row ∈ rows, row < S) (s : State α) :
    ScheduleSafe bounds (rows.map (PairIO.invocation M reuse Y Mean Rstd S D B tree)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rows ih =>
    refine ⟨?_,fun t _ => ih (fun r hr => legal r (List.mem_cons_of_mem _ hr)) t⟩
    have hr := legal row List.mem_cons_self
    apply source_safe M (fun _ _ => Bool.false) reuse (atRow s row)
      "X" "W" "Bias" Y Mean Rstd S D B tree separated bounds
    · intro j hj; exact bounded.2.1 0 (⟨row,hr⟩,⟨j,hj⟩,()) trivial
    · intro j hj; exact bounded.2.1 1 (⟨j,hj⟩,()) trivial
    · intro j hj; exact bounded.2.1 2 (⟨j,hj⟩,()) trivial
    · intro j hj; exact bounded.2.2.1 2 (⟨row,hr⟩,⟨j,hj⟩,()) trivial
    · exact bounded.2.2.1 0 (⟨row,hr⟩,()) trivial
    · exact bounded.2.2.1 1 (⟨row,hr⟩,()) trivial

/-- The full output values and full exterior frame determine every memory cell. -/
theorem memory_unique (M : Algebra α) (s a b : State α)
    (X W Bias Y Mean Rstd : RegionName) (D B : Nat) (tree : ReductionTree B) (rows : List Nat)
    (ha : GridEffect M s a X W Bias Y Mean Rstd D B tree rows)
    (hb : GridEffect M s b X W Bias Y Mean Rstd D B tree rows) : a.mem=b.mem := by
  funext q off
  by_cases hm : ∃ row ∈ rows, q=Mean ∧ off=row
  · obtain ⟨row,hr,hq,ho⟩ := hm
    subst q off
    rw [ha.mean row hr,hb.mean row hr]
  by_cases hi : ∃ row ∈ rows, q=Rstd ∧ off=row
  · obtain ⟨row,hr,hq,ho⟩ := hi
    subst q off
    rw [ha.rstd row hr,hb.rstd row hr]
  by_cases hy : ∃ row ∈ rows, ∃ col, col<D ∧ q=Y ∧ off=col+D*row
  · obtain ⟨row,hr,col,hc,hq,ho⟩ := hy
    subst q off
    rw [ha.output row hr col hc,hb.output row hr col hc]
  have miss : ∀ row ∈ rows, rowMiss Y Mean Rstd D row q off := by
    intro row hr
    refine ⟨?_,?_,?_⟩
    · by_cases hq : q=Mean
      · exact .inr (fun ho => hm ⟨row,hr,hq,ho⟩)
      · exact .inl hq
    · by_cases hq : q=Rstd
      · exact .inr (fun ho => hi ⟨row,hr,hq,ho⟩)
      · exact .inl hq
    · by_cases hq : q=Y
      · exact .inr (fun col hc ho => hy ⟨row,hr,col,hc,hq,ho⟩)
      · exact .inl hq
  rw [ha.frame q off miss,hb.frame q off miss]

theorem complete (M : Algebra α) (reuse : Bool) (s : State α)
    (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd)
    (vals : ChannelValues α (IO.inputs S D)) :
    ∃ final, runSchedule ((PairIO.layerNormIO M id reuse Y Mean Rstd S D c.block c.tree).schedule
      vals) s=some final ∧
      GridEffect M s final "X" "W" "Bias" Y Mean Rstd D c.block c.tree (List.range S) := by
  obtain ⟨t,ht,he⟩ := whole_grid M (fun _ _ => Bool.false) s "X" "W" "Bias" Y Mean Rstd S D
    c.block c.tree reuse c.positive sep (List.range S)
  exact ⟨t,(schedule_eq M reuse Y Mean Rstd S D c.block c.tree (List.range S) s).trans ht,he⟩

/-- Explicit whole-grid semantic preservation from the historical complete
source to either readable program. Equality covers all cells, including the frame. -/
theorem preserve (M : Algebra α) (reuse : Bool) (s : State α)
    (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    ∃ old final,
      Source.launch (withEquality M (fun _ _ => Bool.false)) s Y Mean Rstd S D c.block (SourceBinding.tree c.tree)=some old ∧
      SourceCode.launch (withEquality M (fun _ _ => Bool.false)) reuse s Y Mean Rstd S D c.block (tree c.tree)=some final ∧
      old.mem=final.mem := by
  obtain ⟨old,ho,he⟩ := KernelBoundary.complete M (fun _ _ => Bool.false) s Y Mean Rstd S D c sep
  obtain ⟨final,hf,hg⟩ := whole_grid M (fun _ _ => Bool.false) s "X" "W" "Bias" Y Mean Rstd S D
    c.block c.tree reuse c.positive sep (List.range S)
  exact ⟨old,final,ho,hf,memory_unique M s old final _ _ _ _ _ _ D c.block c.tree _ he hg⟩
#axiomsClean preserve
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
