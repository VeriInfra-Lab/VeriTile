import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Math
namespace VeriTile.Bench.Repairs.Ksoftmax.Counterexamples
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton

theorem causal_log_original :
    Backward.formula 2 0 Bool.true Bool.true (fun _ : Fin 4 => (2 : ℝ))
      (fun _ : Fin 4 => (0 : ℝ)) ⟨1, by decide⟩ = -2 := Backward.causal_log_nonzero_example

theorem causal_log_repaired :
    gradient 2 0 Bool.true Bool.true (fun _ : Fin 4 => (2 : ℝ))
      (fun _ : Fin 4 => (0 : ℝ)) ⟨1, by decide⟩ = 0 :=
  gradient_future 2 0 Bool.true _ _ _ (by decide)

theorem mask_row_address :
    Forward.maskAddress .qk 0 1 1 0=1 ∧ Forward.maskAddress .qk 0 1 3 0=3 := by decide

#axiomsClean causal_log_original
#axiomsClean causal_log_repaired
#axiomsClean mask_row_address
end VeriTile.Bench.Repairs.Ksoftmax.Counterexamples
