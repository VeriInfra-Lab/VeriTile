import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableBoundary
namespace VeriTile.Bench.Repairs.LayerNormWelfold.PairEquivalence
open VeriTile.Triton FP.Structural Reuse
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

private theorem frame (M : Algebra α) (reuse : Bool) (Y Mean Rstd : RegionName)
    (S D : Nat) (c : Configuration) (vals : ChannelValues α (IO.inputs S D))
    (s t : State α)
    (he : GridEffect M s t "X" "W" "Bias" Y Mean Rstd D c.block c.tree (List.range S)) :
    (PairIO.layerNormIO M id reuse Y Mean Rstd S D c.block c.tree).Frame vals s t := by
  intro q off outside
  apply he.frame
  intro row hr
  have legal := List.mem_range.mp hr
  refine ⟨outside.1 0 (⟨row,legal⟩,()) trivial,
    outside.1 1 (⟨row,legal⟩,()) trivial,?_⟩
  by_cases same : q ≠ Y
  · exact .inl same
  · exact .inr (fun col hc => (outside.1 2 (⟨row,legal⟩,⟨col,hc⟩,()) trivial).resolve_left same)

/-- Complete grids and all three output windows are equal for arbitrary
carrier words. The only changed operations are pure integer address computations. -/
theorem equivalent (M : Algebra α) (Y Mean Rstd : RegionName) (S D : Nat)
    (c : Configuration) (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    PairIO.original M Y Mean Rstd S D c.block c.tree ≡[[]]
      PairIO.rowOffsetReused M Y Mean Rstd S D c.block c.tree := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨left,hl,he⟩ := Readable.complete M Bool.false initial Y Mean Rstd S D c sep vals
  obtain ⟨right,hr,hf⟩ := Readable.complete M Bool.true initial Y Mean Rstd S D c sep vals
  have hm := Readable.memory_unique M initial left right "X" "W" "Bias" Y Mean Rstd D c.block c.tree (List.range S) he hf
  refine ⟨?_,?_,left,right,hl,hr,?_,frame M Bool.false Y Mean Rstd S D c vals initial left he,?_⟩
  · exact Readable.schedule_safe M id Bool.false Y Mean Rstd S D c.block c.tree
      (sep.inputs "X" (.inl rfl)).1 vals bounds bounded (List.range S)
      (fun row hr => List.mem_range.mp hr) initial
  · exact Readable.schedule_safe M id Bool.true Y Mean Rstd S D c.block c.tree
      (sep.inputs "X" (.inl rfl)).1 vals bounds bounded (List.range S)
      (fun row hr => List.mem_range.mp hr) initial
  · intro o j _
    change mapCell id (left.mem _ _) = mapCell id (right.mem _ _)
    rw [hm]
  · exact frame M Bool.true Y Mean Rstd S D c vals initial right hf
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.LayerNormWelfold.PairEquivalence
