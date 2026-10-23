import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Grid
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.KernelBinding
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.LogsumexpFwd.KernelResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ApplyPenalty
open MaxReduction.SourceGrid (flatten)
noncomputable section
open Classical

/-- Full successful kernel grid in any complete program order, with every
scratch cell and the entire tagged memory frame. -/
theorem ordered [Inhabited A] (engines : Nat → Engine A) (a : Arguments)
    (scale : A) (s : State A) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    ∃ t,Grid.run engines a scale jobs s=some t ∧
      (∀ j,j<a.rows*a.blocks → t.mem "scratch" j=.mk .real
        (Grid.compute engines a scale (fun i => (s.mem "input" i).read .real) j)) ∧
      (∀ (r : RegionName) i,(r ≠ "scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hm⟩ := Grid.chain engines a scale jobs s
  refine ⟨t,he,?_,?_⟩
  · intro j hj
    have member : j∈jobs := schedule.mem_iff.mpr (List.mem_range.mpr hj)
    have value := Scatter.readback_injective jobs (fun j => (("scratch" : RegionName),j))
      (fun j => FP.Structural.Cell.mk .real
        (Grid.compute engines a scale (fun i => (s.mem "input" i).read .real) j))
      (flatten s) j member (by intro i j h; exact (Prod.mk.inj h).2)
    rw [←hm] at value
    exact value
  · intro r i miss
    change flatten t (r,i)=flatten s (r,i)
    rw [hm]
    apply Scatter.frame
    intro j hj same
    have bound := List.mem_range.mp (schedule.mem_iff.mp hj)
    have region := congrArg Prod.fst same
    have index := congrArg Prod.snd same
    rcases miss with h|h
    · exact h region
    · simp only [Prod.fst,Prod.snd] at index
      omega

theorem local_compute (engines : Nat → Engine A) (a : Arguments) (h : InputContract a)
    (scale : A) (f g : Nat → A) (j : Nat) (hj : j<a.rows*a.blocks)
    (same : ∀ i,i<a.input.numel → f i=g i) :
    Grid.compute engines a scale f j=Grid.compute engines a scale g j := by
  have positive : 0<a.blocks := by nlinarith
  have row : j/a.blocks<a.rows := (Nat.div_lt_iff_lt_mul positive).mpr hj
  apply block_local
  intro i hi
  apply same
  simpa using input_offset_bound a h (j/a.blocks) 0 i row (by simpa using hi)

theorem real_grid (a : Arguments) (h : InputContract a) (scale : ℝ)
    (input : Nat → ℝ) (s : State (WithBot ℝ))
    (binding : ∀ i,i<a.input.numel → (s.mem "input" i).read .real=(input i : WithBot ℝ))
    (jobs : List Nat) (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    ∃ t,Grid.run (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs s=some t ∧
      (∀ row segment,row<a.rows → segment<a.blocks →
        t.mem "scratch" (row*a.blocks+segment)=.mk .real
          (KernelMath.block input row segment a.columns a.block a.hasScale scale : WithBot ℝ)) ∧
      (∀ (r : RegionName) i,(r ≠ "scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hv,hf⟩ := ordered (fun _ => RealModel.engine) a (scale : WithBot ℝ) s jobs schedule
  refine ⟨t,he,?_,hf⟩
  intro row segment hr hb
  have bound := output_offset_bound a row segment hr hb
  have positions := Grid.full_coverage a row segment hr hb
  rw [hv _ bound,local_compute _ a h _ _ _ _ bound binding]
  simp only [Grid.compute,positions.2.1,positions.2.2]
  rw [KernelMath.block_formula _ _ _ _ _ (block_positive a) (active_block a segment hb)]

#axiomsClean ordered
#axiomsClean real_grid
#specNonCircular blockValue avoiding [rowBlock,Source.rowValue,Grid.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.KernelResult
