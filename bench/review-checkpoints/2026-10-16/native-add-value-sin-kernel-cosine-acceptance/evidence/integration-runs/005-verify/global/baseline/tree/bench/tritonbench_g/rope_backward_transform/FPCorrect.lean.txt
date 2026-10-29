import bench.tritonbench_g.rope_transform.FPCorrect
import bench.tritonbench_g.rope_backward_transform.Proofs.TypedShape
namespace VeriTile.Bench.TritonBenchG.RopeBackwardTransform.FPCorrect
open VeriTile.Triton FP.Structural RopeTransform.Typed
noncomputable section
/-- Original backward branch with all four Q/K stores. Every product and
addition/subtraction is a distinct finite-precision backend operation. -/
specification rope_backward_fp32_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (kernel .fp32 .fp32 p) s = some t ∧
      t.mem = referenceMemory M .fp32 .fp32 p s ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) :=
  RopeTransform.FPCorrect.rope_fp32_correctness M p s

specification rope_backward_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (kernel .fp32 .fp32 p) s = some t ∧ t.mem = roundedMemory cfg p s ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) :=
  RopeTransform.FPCorrect.rope_software_fp32 rest cfg p s
#axiomsClean rope_backward_fp32_correctness
#axiomsClean rope_backward_software_fp32
end
end VeriTile.Bench.TritonBenchG.RopeBackwardTransform.FPCorrect
