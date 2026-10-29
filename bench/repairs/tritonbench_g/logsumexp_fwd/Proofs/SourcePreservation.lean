import bench.repairs.tritonbench_g.logsumexp_fwd.Kernels
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.LegacyPastedSurface

/-! Infinity is compiled by the same existing exact word-literal primitive.
Only immutable cdiv arithmetic in the final address has a different AST; its
complete execution is proved equal below for every backend and initial state. -/
namespace VeriTile.Bench.Repairs.LogsumexpFwd.SourcePreservation
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem source_prefix (x z : RegionName) (D B : Nat) (scaled : Bool) :
    (rowBlock x z D B scaled).surfaceBody.take 10 =
      (LegacyPastedSurface.rowBlock x z D B scaled).surfaceBody.take 10 := rfl

theorem candidate_prefix (x z : RegionName) (D B : Nat) (scaled : Bool) :
    (rowBlockCandidate x z D B scaled).surfaceBody.take 9 =
      (LegacyPastedSurface.rowBlockCandidate x z D B scaled).surfaceBody.take 9 := rfl

private theorem tail_run {A : Type} [Inhabited A] (M : Algebra A)
    (x z : RegionName) (D B : Nat) (scaled : Bool) (s : State A) :
    run M ((rowBlock x z D B scaled).surfaceBody.drop 10) s =
      run M ((LegacyPastedSurface.rowBlock x z D B scaled).surfaceBody.drop 10) s := by
  simp [rowBlock,LegacyPastedSurface.rowBlock,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,numeric,bop,
    ComputeDType.eraseDType,Region.cast]

theorem original {A : Type} [Inhabited A] (M : Algebra A)
    (x z : RegionName) (D B : Nat) (scaled : Bool) (s : State A) :
    exec M (rowBlock x z D B scaled) s =
      exec M (LegacyPastedSurface.rowBlock x z D B scaled) s := by
  change run M ((rowBlock x z D B scaled).surfaceBody.take 10 ++
      (rowBlock x z D B scaled).surfaceBody.drop 10) s =
    run M ((LegacyPastedSurface.rowBlock x z D B scaled).surfaceBody.take 10 ++
      (LegacyPastedSurface.rowBlock x z D B scaled).surfaceBody.drop 10) s
  rw [run_append,run_append,source_prefix]
  congr 1
  funext t
  exact tail_run M x z D B scaled t

theorem candidate {A : Type} [Inhabited A] (M : Algebra A)
    (x z : RegionName) (D B : Nat) (scaled : Bool) (s : State A) :
    exec M (rowBlockCandidate x z D B scaled) s =
      exec M (LegacyPastedSurface.rowBlockCandidate x z D B scaled) s := by
  change run M ((rowBlockCandidate x z D B scaled).surfaceBody.take 9 ++
      (rowBlockCandidate x z D B scaled).surfaceBody.drop 9) s =
    run M ((LegacyPastedSurface.rowBlockCandidate x z D B scaled).surfaceBody.take 9 ++
      (LegacyPastedSurface.rowBlockCandidate x z D B scaled).surfaceBody.drop 9) s
  rw [run_append,run_append,candidate_prefix]
  congr 1
  funext t
  exact tail_run M x z D B scaled t

#axiomsClean source_prefix
#axiomsClean candidate_prefix
#axiomsClean original
#axiomsClean candidate
end
end VeriTile.Bench.Repairs.LogsumexpFwd.SourcePreservation
