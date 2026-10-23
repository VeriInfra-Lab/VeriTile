import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.LogsumexpFwd.RawReal
open VeriTile.Triton FP.Structural
noncomputable section

/-- All independent block logsumexp values, complete quoted-source grid
execution in any order, and every exterior cell on raw logical input memory. -/
theorem logsumexp_real_raw (a : RawKernel.Geometry) (scale : ℝ)
    (input : Nat → ℝ) (s : State (WithBot ℝ))
    (binding : ∀ row,row<a.rows → ∀ col,col<a.columns →
      (s.mem "input" (row*a.columns+col)).read .real=(input (row*a.columns+col) : WithBot ℝ))
    (jobs : List Nat) (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    VeriTile.Spec.Real (∃ t,
      RawKernel.run (fun _=>RealModel.engine) a (scale : WithBot ℝ) jobs s=some t ∧
      (∀ row segment,row<a.rows → segment<a.blocks →
        t.mem "scratch" (row*a.blocks+segment)=.mk .real
          (KernelMath.block input row segment a.columns a.block a.hasScale scale : WithBot ℝ)) ∧
      (∀ (r : RegionName) i,(r≠"scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i)) := by
  obtain ⟨t,ran,values,frame⟩ := RawKernel.ordered (fun _=>RealModel.engine) a (scale : WithBot ℝ) s jobs schedule
  refine ⟨t,ran,?_,frame⟩
  intro row segment hr hb
  have bound := RawKernel.output_offset_bound a row segment hr hb
  have positions := RawKernel.full_coverage a row segment hr hb
  rw [values _ bound]
  simp only [RawKernel.compute,positions.2.1,positions.2.2]
  rw [block_local _ _ (fun i=>(input i : WithBot ℝ)) _ _ _ _ _ _ (binding row hr)]
  rw [KernelMath.block_formula _ _ _ _ _ a.positive (RawKernel.active_block a segment hb)]

#axiomsClean logsumexp_real_raw
#specNonCircular KernelMath.block avoiding [rowBlock,Source.rowValue,RawKernel.run]
end
end VeriTile.Bench.Repairs.LogsumexpFwd.RawReal
