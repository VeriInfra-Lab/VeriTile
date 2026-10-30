import bench.tritonbench_g.logsumexp_fwd.Proofs.Interface

namespace VeriTile.Bench.Repairs.LogsumexpFwd.Interface
open VeriTile.Triton FP.Structural
noncomputable section

theorem real_implements (a : RawKernel.Geometry) (scale : ℝ) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (a.rows * a.blocks))) :
    (programIO (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs).ImplementsOn
      (RealInputs a) (fun values => result (fun j =>
        (KernelMath.block (realInput values) (j.val / a.blocks) (j.val % a.blocks)
          a.columns a.block a.hasScale scale : WithBot ℝ))) := by
  intro values domain bounds hb s hin
  obtain ⟨safe, t, ran, outputs, frame⟩ :=
    implements (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs schedule
      values trivial bounds hb s hin
  refine ⟨safe, t, ran, ?_, frame⟩
  intro o j active
  fin_cases o
  have bound : j.1.val < a.rows * a.blocks := j.1.isLt
  have positive : 0 < a.blocks := by nlinarith
  have row : j.1.val / a.blocks < a.rows := (Nat.div_lt_iff_lt_mul positive).mpr bound
  have segment : j.1.val % a.blocks < a.blocks := Nat.mod_lt _ positive
  have target : RawKernel.compute (fun _ => RealModel.engine) a (scale : WithBot ℝ)
      (extend values) j.1.val =
      (KernelMath.block (realInput values) (j.1.val / a.blocks) (j.1.val % a.blocks)
        a.columns a.block a.hasScale scale : WithBot ℝ) := by
    unfold RawKernel.compute
    rw [block_local _ _ (fun i => (realInput values i : WithBot ℝ))]
    · exact KernelMath.block_formula _ _ _ _ _ a.positive (RawKernel.active_block a _ segment) _ _
    · intro i hi
      have index : j.1.val / a.blocks * a.columns + i < a.rows * a.columns := by nlinarith
      obtain ⟨x, hx⟩ := domain ⟨_, index⟩
      simp [extend, realInput, index, hx]
      rfl
  simpa [result, target] using outputs ⟨0, by simp [programIO, memory, Interface.outputs]⟩ j active

#axiomsClean real_implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.Interface
