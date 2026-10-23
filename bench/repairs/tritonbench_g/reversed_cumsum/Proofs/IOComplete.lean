import bench.repairs.tritonbench_g.reversed_cumsum.IO
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.ReversedCumsum.IOComplete
open VeriTile.Triton Interface
noncomputable section

@[simp] theorem lane_time (h : Call) (t : Fin h.T) (j : Fin BS) :
    (lane h t j).val/BS=t.val := by
  have hj := j.isLt
  dsimp [lane,BS] at *
  omega

@[simp] theorem lane_feature (h : Call) (t : Fin h.T) (j : Fin BS) :
    (lane h t j).val%BS=j.val := by
  simp [lane,Nat.add_mod,Nat.mod_eq_of_lt j.isLt]

theorem formula (h : Call) (program : Geometry.Program h) (xs : Nat → ℝ) (vals : Inputs h)
    (j : Fin (h.T*BS)) (active : program.2.val*BS+j.val%BS<h.S)
    (hin : ∀ (t : Fin h.T) (k : Fin BS),program.2.val*BS+k.val<h.S →
      xs (Geometry.address h program.1.val t.val (program.2.val*BS+k.val))=vals 0 (lane h t k)) :
    RealExecution.suffix h xs program.1.val (j.val/BS) (program.2.val*BS+j.val%BS)=
      reference h vals 0 j := by
  have jt : j.val/BS<h.T := (Nat.div_lt_iff_lt_mul (by decide : 0<BS)).mpr j.isLt
  unfold RealExecution.suffix
  rw [VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffix_filter _ _ _ (Nat.le_of_lt jt)]
  unfold reference
  rw [Finset.sum_filter,Finset.sum_range]
  apply Finset.sum_congr rfl
  intro t _
  rw [hin t ⟨j.val%BS,Nat.mod_lt _ (by decide)⟩ active]

theorem implements (h : Call) (distinct : h.input≠h.output) (c : Config)
    (chosen : c∈configurations) (program : Geometry.Program h) :
    (io h c program).Implements (reference h) := by
  intro vals _ initial hin
  let entered := (launch program).initial initial
  have positive := (Kernels.configurations_positive c chosen).1
  obtain ⟨final,ran,values,frame⟩ := RealExecution.source_correct h c program.1.val program.2
    entered distinct positive (by simp [entered,launch,AlgorithmLaunch.initial])
    (by simp [entered,launch,AlgorithmLaunch.initial])
  let trace : AlgorithmExecution ((io h c program).stages vals) initial final :=
    .cons ((AlgorithmStage.single_runs _ _ _ _).mpr ran) (.nil final)
  refine ⟨final,trace,?_,?_,?_⟩
  · intro bounds backed
    refine ⟨?_,trivial⟩
    change Kernel.TraceSafe bounds (Kernels.source h c).eraseDType.toAlgKernel entered
    apply SourceSafety.source_safe h c program.1.val program.2 entered distinct positive
      (by simp [entered,launch,AlgorithmLaunch.initial]) (by simp [entered,launch,AlgorithmLaunch.initial]) bounds
    constructor
    · intro t j ht active
      have v := backed.1 (0 : Fin 1) (lane h ⟨t,ht⟩ j)
      simp only [io,lane_time,lane_feature] at v
      exact v active
    · intro t j ht active
      have v := backed.2.1 (0 : Fin 1) (lane h ⟨t,ht⟩ j)
      simp only [io,lane_time,lane_feature] at v
      exact v active
  · intro port j active
    fin_cases port
    change final.readMem h.output
      (Geometry.address h program.1.val (j.val/BS) (program.2.val*BS+j.val%BS))=_
    have jt : j.val/BS<h.T := (Nat.div_lt_iff_lt_mul (by decide : 0<BS)).mpr j.isLt
    have value := values (j.val/BS) ⟨j.val%BS,Nat.mod_lt _ (by decide)⟩ jt active
    rw [BlockState.readMem,value]
    change RealExecution.suffix h (entered.readMem h.input) program.1.val (j.val/BS)
      (program.2.val*BS+j.val%BS)=_
    apply formula h program _ vals j active
    intro t k hk
    have v := hin (0 : Fin 1) (lane h t k)
    simp only [io,lane_time,lane_feature] at v
    exact v hk
  · intro r o outside _
    have no : ¬RealExecution.footprint h program.1.val program.2.val (r,o) := by
      rintro ⟨same,t,j,⟨ht,hj⟩,eq⟩
      have conflict := outside (0 : Fin 1) (lane h ⟨t,ht⟩ j)
      have hj' : program.2.val*BS+j.val<h.S := hj
      have conflict' : r≠h.output ∨ o≠Geometry.address h program.1.val t (program.2.val*BS+j.val) := by
        simp only [io,lane_time,lane_feature] at conflict
        exact conflict hj'
      have eq' : o=Geometry.address h program.1.val t (program.2.val*BS+j.val) := by
        simpa [VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address,RealProgram.layout,
          RealProgram.launch,Geometry.address] using eq
      exact conflict'.elim (fun hn => hn same) (fun hn => hn eq')
    exact (frame r o no).symm

#axiomsClean implements
end
end VeriTile.Bench.Repairs.ReversedCumsum.IOComplete
