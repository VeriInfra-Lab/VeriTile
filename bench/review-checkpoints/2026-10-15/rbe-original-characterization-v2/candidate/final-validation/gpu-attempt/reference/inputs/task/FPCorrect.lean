import bench.tritonbench_g.rbe_triton_transform.Proofs.FPWitness
import bench.tritonbench_g.rbe_triton_transform.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.RbeTritonTransform
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete accepted pair surface: actual fp16 storage, fp32 computation scopes,
both masked outputs, initial-state reference and frame. The numerical algebra
is still an explicit parameter: native transcendental/mixed-cast binding remains
open; this is not a closed software bit theorem. -/
specification rbe_triton_backend_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : FPTyped.Parameters) (s : State α) (hw : FPTyped.WritesApart p s) :
    ∃ final, FP.Structural.exec M (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPTyped.value M p s) :=
  FPTyped.correctness M p s hw

theorem rbe_triton_source_projection (p : FPTyped.Parameters) :
    (FPTyped.source p).eraseDType.surfaceBody = (FPTyped.legacySource p).eraseDType.surfaceBody :=
  FPTyped.source_projection p

theorem rbe_triton_backend_success {α : Type} [Inhabited α] (M : Algebra α) (a b sentinel : α) :
    ∃ final, FP.Structural.exec M (FPTyped.source FPTyped.witnessParameters)
      (FPTyped.witnessInitial a b sentinel) = some final ∧
      final.mem "output" 0 = .mk .fp16
        (FPTyped.value M FPTyped.witnessParameters (FPTyped.witnessInitial a b sentinel) Bool.false FPTyped.lane) ∧
      final.mem "output" 1 = .mk .fp16
        (FPTyped.value M FPTyped.witnessParameters (FPTyped.witnessInitial a b sentinel) Bool.true FPTyped.lane) ∧
      final.mem "input" 0 = .mk .fp16 a ∧ final.mem "output" 2 = .mk .fp16 sentinel :=
  FPTyped.nonempty_backend M a b sentinel

#axiomsClean rbe_triton_backend_correctness
#axiomsClean rbe_triton_source_projection
#axiomsClean rbe_triton_backend_success
end
end VeriTile.Bench.TritonBenchG.RbeTritonTransform
