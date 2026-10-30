import bench.tritonbench_g.reversed_cumsum_scalar.IO
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.SourceSafety
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.KernelCompatibility

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.IOComplete
open VeriTile.Triton Interface
noncomputable section

theorem formula (h : Call) (xs : Nat → ℝ) (vals : Inputs h)
    (hin : ∀ t : Fin h.T,xs t.val=vals 0 t) (j : Fin h.T) :
    VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix xs h.T j.val =
      reference h vals 0 j := by
  rw [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix_filter xs h.T j.val (Nat.le_of_lt j.isLt)]
  unfold reference
  rw [Finset.sum_filter,Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  rw [hin k]

theorem implements (h : Call) (layout : Geometry.OutputLayout h) (c : Config)
    (chosen : c∈configurations) (row : Fin (h.B*h.H)) :
    (io h c row).Implements (reference h) := by
  intro vals _ initial hin
  let entered := (launch row).initial initial
  have hr : entered.pids 0<h.B*h.H := row.isLt
  have positive := (Kernels.configurations_positive c chosen).1
  obtain ⟨final,ran,values,frame⟩ := RealExecution.source_correct h layout c positive entered hr
  let trace : AlgorithmExecution ((io h c row).stages vals) initial final :=
    .cons ((AlgorithmStage.single_runs _ _ _ _).mpr ran) (.nil final)
  refine ⟨final,trace,?_,?_,?_⟩
  · intro bounds backed
    refine ⟨?_,trivial⟩
    change Kernel.TraceSafe bounds (Kernels.source h c).eraseDType.toAlgKernel entered
    apply SourceSafety.source_safe h layout c positive entered hr bounds
    · intro t ht
      exact backed.1 (0 : Fin 1) ⟨t,ht⟩ trivial
    · intro t ht
      exact backed.2.1 (0 : Fin 1) ⟨t,ht⟩ trivial
  · intro port j _
    fin_cases port
    change final.readMem h.output (h.outputStrides.address h.H row.val j.val)=_
    have value := values j.val j.isLt
    change final.mem h.output (h.outputStrides.address h.H row.val j.val)=_ at value
    rw [BlockState.readMem,value]
    change VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix
      (fun k => entered.readMem h.input (h.inputStrides.address h.H row.val k)) h.T j.val = _
    apply formula h _ vals _ j
    intro t
    exact hin (0 : Fin 1) t trivial
  · intro r o outside _
    have no : ¬RealMemory.footprint h row.val (r,o) := by
      rintro ⟨same,t,ht,eq⟩
      exact (outside (0 : Fin 1) ⟨t,ht⟩ trivial).elim (fun hn => hn same) (fun hn => hn eq)
    exact (frame r o no).symm

end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.IOComplete
