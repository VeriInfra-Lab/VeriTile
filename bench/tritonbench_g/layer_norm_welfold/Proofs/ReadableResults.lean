import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableBoundary
import bench.tritonbench_g.layer_norm_welfold.Proofs.KernelRealResult
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

/-- Transport an existing formula only after proving complete old/new memory
preservation and independent safety for the actual readable source. -/
theorem implements (M : Algebra α) (observe : α → α) (reuse : Bool)
    (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd)
    (f : ChannelValues α (IO.inputs S D) → ChannelValues α (IO.outputs Y Mean Rstd S D))
    (old : (IO.layerNormIO M observe Y Mean Rstd S D c.block c.tree).Implements f) :
    (PairIO.layerNormIO M observe reuse Y Mean Rstd S D c.block c.tree).Implements f := by
  intro vals _ bounds bounded initial binding
  obtain ⟨_,oldFinal,oldRan,values,frame⟩ := old vals trivial bounds bounded initial binding
  obtain ⟨historical,final,ho,hf,hm⟩ := preserve M reuse initial Y Mean Rstd S D c sep
  have histRan : runSchedule ((IO.layerNormIO M observe Y Mean Rstd S D c.block c.tree).schedule vals) initial=some historical :=
    (IOProof.schedule_eq M Y Mean Rstd S D c.block c.tree (List.range S) initial).trans ho
  have eq : oldFinal=historical := Option.some.inj (oldRan.symm.trans histRan)
  subst oldFinal
  refine ⟨schedule_safe M observe reuse Y Mean Rstd S D c.block c.tree
      (sep.inputs "X" (.inl rfl)).1 vals bounds bounded (List.range S)
      (fun row hr => List.mem_range.mp hr) initial,final,?_,?_,?_⟩
  · exact (schedule_eq M reuse Y Mean Rstd S D c.block c.tree (List.range S) initial).trans hf
  · intro o j active
    change mapCell observe (final.mem _ _) = _
    rw [←hm]
    exact values o j active
  · change (IO.layerNormIO M observe Y Mean Rstd S D c.block c.tree).Frame vals initial final
    unfold ProgramIO.Frame at frame ⊢
    simpa only [←hm] using frame

def realFormula (D : Nat) (x w bias : Nat → ℝ) (row : Nat) : ℝ × ℝ × (Nat → ℝ) :=
  (Math.mean x D row,Math.inverse x D row,Math.output x w bias D row)

theorem real (rest : Algebra ℝ) (Y Mean Rstd : RegionName) (S D : Nat)
    (c : Configuration) (complete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    (PairIO.layerNormIO (Samples.ideal rest) id Bool.false Y Mean Rstd S D c.block c.tree).Implements
      (IO.result Y Mean Rstd S D (realFormula D)) :=
by
  apply implements _ _ _ _ _ _ _ _ _ sep
  apply IOProof.implements _ _ _ _ _ _ _ _ _ (sep.inputs "X" (Or.inl rfl)).1
  intro initial x w bias hx hw hb
  obtain ⟨final,ran,hm,hi,hy,frame⟩ := KernelRealResult.kernel_real rest initial x w bias
    S D Y Mean Rstd c complete sep hx hw hb
  refine ⟨final,ran,?_,?_,?_,frame⟩
  · intro row hr; rw [hm row hr]; rfl
  · intro row hr; rw [hi row hr]; rfl
  · intro row hr col hc; rw [hy row hr col hc]; rfl
#axiomsClean real
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
