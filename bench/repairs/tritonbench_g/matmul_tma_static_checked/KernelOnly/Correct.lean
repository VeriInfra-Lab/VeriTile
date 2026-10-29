import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
open VeriTile.Triton FP.Structural FPValue Host TensorABI
noncomputable section

specification kernel_real (rest : Algebra ℝ) (read : Nat → ℝ) (h : Shape) (f : Flags) (cfg : Config)
    (_checked : InputContract h cfg) (a b c : TensorABI.Tensor) (abi : Binding h f a b c)
    (s : State ℝ) (input : Inputs read h a b s) :
    VeriTile.Spec.Real (∃ t, exec (backend (RealModel.algebra rest) RealModel.dot) (kernel h f) s=some t ∧
      Post (params h f) f.half s t (RealModel.product read h f a b) ∧
      RealModel.TensorPost read (RealModel.render read h f c t) h f a b c) := by
  obtain ⟨t,he,_,post⟩ := executes (RealModel.algebra rest) RealModel.dot h f s
  rw [RealModel.output_product rest read h f a b c abi s input] at post
  exact ⟨t,he,post,RealModel.tensor_post read h f a b c abi s t post⟩

#axiomsClean kernel_real
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
